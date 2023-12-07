use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, OnceLock};

use regex::{Error as RegexError, Regex};

use crate::bytes_util::FastRegexMatcher;
use crate::regex_util::match_handlers::StringMatchHandler;
use crate::regex_util::prefix_cache::{PrefixCache, PrefixSuffix};
use crate::regex_util::regex_utils::{FULL_MATCH_COST, LITERAL_MATCH_COST};
use crate::regex_util::regexp_cache::{RegexpCache, RegexpCacheValue};
use crate::regex_util::{get_match_func_for_or_suffixes, get_optimized_re_match_func, regex_utils};

/// TagFilters represents filters used for filtering tags.
#[derive(Clone, Default, Debug)]
pub struct TagFilters(pub Vec<TagFilter>);

impl TagFilters {
    pub fn new(filters: Vec<TagFilter>) -> Self {
        let mut filters = filters;
        filters.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self(filters)
    }
    pub fn is_match(&self, b: &str) -> bool {
        // todo: should sort first
        self.0.iter().all(|tf| tf.is_match(b))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&TagFilter> {
        self.0.get(index)
    }
    pub fn sort(&mut self) {
        self.0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    /// Adds the given tag filter.
    ///
    /// metric_group must be encoded with nil key.
    pub fn add(
        &mut self,
        key: &str,
        value: &str,
        is_negative: bool,
        is_regexp: bool,
    ) -> Result<(), String> {
        let mut is_negative = is_negative;
        let mut is_regexp = is_regexp;

        let mut value_ = value;
        // Verify whether tag filter is empty.
        if value.is_empty() {
            // Substitute an empty tag value with the negative match of `.+` regexp in order to
            // filter out all the values with the given tag.
            is_negative = !is_negative;
            is_regexp = true;
            value_ = ".+";
        }
        if is_regexp && value == ".*" {
            if !is_negative {
                // Skip tag filter matching anything, since it equals to no filter.
                return Ok(());
            }

            // Substitute negative tag filter matching anything with negative tag filter matching non-empty value
            // in order to filter out all the time series with the given key.
            value_ = ".+";
        }

        let tf = TagFilter::new(key, value_, is_negative, is_regexp)
            .map_err(|err| format!("cannot parse tag filter: {}", err))?;

        if tf.is_negative && tf.is_empty_match {
            // We have {key!~"|foo"} tag filter, which matches non-empty key values.
            // So add {key=~".+"} tag filter in order to enforce this.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/546 for details.
            let tf_new = TagFilter::new(key, ".+", false, true)
                .map_err(|err| format!("cannot parse tag filter: {}", err))?;

            self.0.push(tf_new);
        }

        self.0.push(tf);
        Ok(())
    }

    /// Reset resets the tf
    pub(crate) fn reset(&mut self) {
        self.0.clear();
    }
}

impl Display for TagFilters {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let a = self
            .0
            .iter()
            .map(|tf| tf.to_string())
            .collect::<Vec<String>>();
        write!(f, "{:?}", a)
    }
}

/// TagFilter represents a filter used for filtering tags.
#[derive(Clone, Default, Debug)]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub is_negative: bool,
    pub is_regexp: bool,
    pub is_literal: bool,

    /// match_cost is a cost for matching a filter against a single string.
    pub match_cost: usize,

    /// Prefix contains
    ///  - value if !is_regexp.
    ///  - regexp_prefix if is_regexp.
    pub prefix: String,
    pub(crate) prefix_match: StringMatchHandler,

    /// `or` values obtained from regexp suffix if it equals to "foo|bar|..."
    ///
    /// the regexp prefix is stored in regexp_prefix.
    ///
    /// This array is also populated with matching Graphite metrics if key="__graphite__"
    pub or_suffixes: Vec<String>,

    /// Matches suffix.
    pub(crate) suffix_match: StringMatchHandler,

    /// Set to true for filters matching empty value.
    pub is_empty_match: bool,

    /// Contains reverse suffix for Graphite wildcard.
    /// I.e. for `{__name__=~"foo\\.[^.]*\\.bar\\.baz"}` the value will be `zab.rab.`
    pub graphite_reverse_suffix: String,
}

impl TagFilter {
    /// creates the tag filter for the given common_prefix, key and value.
    ///
    /// If is_negative is true, then the tag filter matches all the values except the given one.
    ///
    /// If is_regexp is true, then the value is interpreted as anchored regexp, i.e. '^(tag.Value)$'.
    ///
    /// MetricGroup must be encoded in the value with nil key.
    pub fn new(
        key: &str,
        value: &str,
        is_negative: bool,
        is_regexp: bool,
    ) -> Result<TagFilter, String> {
        let mut tf = TagFilter::default();
        tf.key = key.to_string();
        tf.value = value.to_string();
        tf.is_negative = is_negative;
        tf.is_regexp = is_regexp;
        tf.match_cost = 0;

        tf.is_empty_match = false;

        tf.prefix = key.to_string();

        let prefix = value.to_string();
        if tf.is_regexp {
            let (prefix, expr) = simplify_regexp(value)
                .map_err(|err| format!("cannot simplify regexp {}: {}", value, err))?;

            if expr.is_empty() {
                tf.value = prefix.clone();
                tf.is_regexp = false;
                tf.is_literal = true;
                tf.prefix_match = if tf.is_negative {
                    StringMatchHandler::literal_mismatch(prefix)
                } else {
                    StringMatchHandler::literal(prefix)
                };
            }
        }
        tf.prefix = prefix.to_string();
        if !tf.is_regexp {
            // tf contains plain value without regexp.
            // Add empty or_suffix in order to trigger fast path for or_suffixes during the search for
            // matching metricIDs.
            tf.or_suffixes.push("".to_string());
            tf.is_empty_match = prefix.len() == 0;
            tf.match_cost = FULL_MATCH_COST;
            return Ok(tf);
        }
        let rcv = get_regexp_from_cache(value)?;
        tf.or_suffixes = rcv.or_values.clone();
        tf.suffix_match = rcv.re_match.clone();
        tf.match_cost = rcv.re_cost;
        tf.is_empty_match = prefix.len() == 0 && tf.suffix_match.matches("");
        if !tf.is_negative && key.is_empty() && rcv.as_ref().literal_suffix.contains('.') {
            // Reverse suffix is needed only for non-negative regexp filters on __name__ that contains dots.
            tf.graphite_reverse_suffix = rcv.literal_suffix.chars().rev().collect::<String>();
        }
        Ok(tf)
    }

    pub fn is_match(&self, b: &str) -> bool {
        let good = self.prefix_match.matches(b);
        if !good || self.is_literal {
            return good;
        }
        let prefix = &self.prefix;
        let ok = self.match_suffix(&b[prefix.len()..]);
        if !ok {
            return self.is_negative;
        }
        return !self.is_negative;
    }

    #[inline]
    pub fn match_suffix(&self, b: &str) -> bool {
        if !self.is_regexp {
            return b.len() == 0;
        }
        self.suffix_match.matches(b)
    }

    pub fn get_op(&self) -> &'static str {
        if self.is_negative {
            if self.is_regexp {
                return "!~";
            }
            return "!=";
        }
        if self.is_regexp {
            return "=~";
        }
        return "=";
    }
}

impl PartialEq<Self> for TagFilter {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for TagFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.match_cost != other.match_cost {
            return Some(self.match_cost.cmp(&other.match_cost));
        }
        if self.is_regexp != other.is_regexp {
            return Some(self.is_regexp.cmp(&other.is_regexp));
        }
        if self.or_suffixes.len() != other.or_suffixes.len() {
            return Some(self.or_suffixes.len().cmp(&other.or_suffixes.len()));
        }
        if self.is_negative != other.is_negative {
            return Some(self.is_negative.cmp(&other.is_negative));
        }
        return Some(self.prefix.cmp(&other.prefix));
    }
}

// String returns human-readable tf value.
impl Display for TagFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let op = self.get_op();
        let value = if self.value.len() > 60 {
            // todo: could panic for non-ascii
            &self.value[0..60]
        } else {
            &self.value
        };

        if self.key.len() == 0 {
            return write!(f, "__name__{op}{value}");
        }
        write!(f, "{}{}{}", self.key, op, value)
    }
}

pub(super) fn compile_regexp(expr: &str) -> Result<RegexpCacheValue, String> {
    // Slow path - build the regexp.
    let expr_str = format!("^({expr})$");
    let re =
        Regex::new(&expr_str).map_err(|e| format!("cannot compile regexp {}: {}", expr_str, e))?;

    let or_values = regex_utils::get_or_values(expr);

    let (re_match, literal_suffix, re_cost) = if !or_values.is_empty() {
        let (match_fn, re_cost) = new_match_func_for_or_suffixes(or_values.clone());
        (match_fn, "".to_string(), re_cost)
    } else {
        let matcher = StringMatchHandler::FastRegex(FastRegexMatcher::new(re));
        let (re_match, literal_suffix, re_cost) = get_optimized_re_match_func(matcher, expr);
        (re_match, literal_suffix, re_cost)
    };

    // heuristic for rcv in-memory size
    let size_bytes = 8 * expr.len() + literal_suffix.len() + 8 * or_values.len();

    // Put the re_match in the cache.
    Ok(RegexpCacheValue {
        re_match,
        re_cost,
        literal_suffix,
        or_values,
        size_bytes,
    })
}

pub fn get_regexp_from_cache(expr: &str) -> Result<Arc<RegexpCacheValue>, String> {
    let cache = get_regexp_cache();
    if let Some(rcv) = cache.get(expr) {
        // Fast path - the regexp found in the cache.
        return Ok(rcv);
    }

    // Put the re_match in the cache.
    let rcv = compile_regexp(expr)?;
    Ok(cache.put(expr, rcv))
}

fn new_match_func_for_or_suffixes(or_values: Vec<String>) -> (StringMatchHandler, usize) {
    let re_cost = or_values.len() * LITERAL_MATCH_COST;
    let matcher = get_match_func_for_or_suffixes(or_values);
    return (matcher, re_cost);
}

const DEFAULT_MAX_REGEXP_CACHE_SIZE: usize = 2048;
const DEFAULT_MAX_PREFIX_CACHE_SIZE: usize = 2048;

fn get_regexp_cache_max_size() -> &'static usize {
    static REGEXP_CACHE_MAX_SIZE: OnceLock<usize> = OnceLock::new();
    REGEXP_CACHE_MAX_SIZE.get_or_init(|| {
        // todo: read value from env
        let size = DEFAULT_MAX_REGEXP_CACHE_SIZE;
        size
    })
}

fn get_prefix_cache_max_size() -> &'static usize {
    static REGEXP_CACHE_MAX_SIZE: OnceLock<usize> = OnceLock::new();
    REGEXP_CACHE_MAX_SIZE.get_or_init(|| {
        // todo: read value from env
        let size = DEFAULT_MAX_PREFIX_CACHE_SIZE;
        size
    })
}

static REGEX_CACHE: OnceLock<RegexpCache> = OnceLock::new();
static PREFIX_CACHE: OnceLock<PrefixCache> = OnceLock::new();

// todo: get from env

pub fn get_regexp_cache() -> &'static RegexpCache {
    REGEX_CACHE.get_or_init(|| {
        let size = get_regexp_cache_max_size();
        let cache = RegexpCache::new(*size);
        cache
    })
}

pub fn get_prefix_cache() -> &'static PrefixCache {
    PREFIX_CACHE.get_or_init(|| {
        let size = *get_prefix_cache_max_size();
        let cache = PrefixCache::new(size);
        cache
    })
}

pub fn simplify_regexp(expr: &str) -> Result<(String, String), RegexError> {
    let cache = get_prefix_cache();
    if let Some(ps) = cache.get(expr) {
        // Fast path - the simplified expr is found in the cache.
        return Ok((ps.prefix.clone(), ps.suffix.clone()));
    }

    // Slow path - simplify the expr.

    // Make a copy of expr before using it,
    let (prefix, suffix) = regex_utils::simplify(expr)?;

    // Put the prefix and the suffix to the cache.
    let ps = PrefixSuffix {
        prefix: prefix.clone(),
        suffix: suffix.clone(),
    };
    let _ = cache.put(expr, ps);

    Ok((prefix, suffix))
}
