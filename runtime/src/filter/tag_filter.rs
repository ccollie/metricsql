use crate::prelude::METRIC_NAME_LABEL;
use crate::RuntimeResult;
use metricsql_common::regex_util::FULL_MATCH_COST;
use metricsql_common::regex_util::{compile_regexp_anchored, StringMatchHandler};
use metricsql_parser::label::{LabelFilter, LabelFilterOp, Matchers};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};


/// TagFilter represents a filter used for filtering tags.
#[derive(Clone, Default, Debug)]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub is_negative: bool,
    pub is_regexp: bool,

    /// match_cost is a cost for matching a filter against a single string.
    pub match_cost: usize,
    pub matcher: StringMatchHandler,

    /// Set to true for filters matching empty value.
    pub is_empty_match: bool,
}

impl TagFilter {
    pub fn from_label_filter(filter: &LabelFilter) -> Result<Self, String> {
        match filter.op {
            LabelFilterOp::Equal=> {
                Self::new(&filter.label, &filter.value, false, false)
            }
            LabelFilterOp::NotEqual => {
                Self::new(&filter.label, &filter.value, true, false)
            }
            LabelFilterOp::RegexEqual => {
                Self::new(&filter.label, &filter.value, false, true)
            }
            LabelFilterOp::RegexNotEqual => {
                Self::new(&filter.label, &filter.value, true, true)
            }
        }
    }

    /// creates the tag filter for the given common_prefix, key and value.
    ///
    /// If is_negative is true, then the tag filter matches all the values except the given one.
    ///
    /// If is_regexp is true, then the value is interpreted as anchored regexp, i.e. '^(tag.Value)$'.
    pub fn new(
        key: &str,
        value: &str,
        is_negative: bool,
        is_regexp: bool,
    ) -> Result<TagFilter, String> {

        let (matcher, match_cost) = if is_regexp {
            compile_regexp_anchored(value)?
        } else {
            (StringMatchHandler::Literal(key.to_string()), FULL_MATCH_COST)
        };
        // tf.is_empty_match = prefix.is_empty() && tf.suffix_match.matches("");

        Ok(TagFilter {
            key: key.to_string(),
            value: value.to_string(),
            is_negative,
            is_regexp,
            match_cost,
            matcher,
            is_empty_match: false,
        })
    }

    pub fn matches(&self, b: &str) -> bool {
        let good = self.matcher.matches(b);
        if self.is_negative {
            !good
        } else {
            good
        }
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
        "="
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
        if self.is_negative != other.is_negative {
            return Some(self.is_negative.cmp(&other.is_negative));
        }
        Some(Ordering::Equal)
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

        if self.key.is_empty() {
            return write!(f, "{METRIC_NAME_LABEL}{op}{value}");
        }
        write!(f, "{}{}{}", self.key, op, value)
    }
}

/// TagFilters represents filters used for filtering tags. All the filters must match for a tag to pass.
#[derive(Clone, Default, Debug)]
pub struct TagFilters(pub Vec<TagFilter>);

impl TagFilters {
    pub fn new(filters: Vec<TagFilter>) -> Self {
        let mut filters = filters;
        filters.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self(filters)
    }

    pub fn add_label_filters(&mut self, filters: &[LabelFilter]) -> Result<(), String> {
        for filter in filters.iter() {
            self.add_label_filter(filter)?;
        }
        self.sort();
        Ok(())
    }

    pub fn add_label_filter(&mut self, filter: &LabelFilter) -> Result<(), String> {
        match filter.op {
            LabelFilterOp::Equal=> {
                self.add(&filter.label, &filter.value, false, false)
            }
            LabelFilterOp::NotEqual => {
                self.add(&filter.label, &filter.value, true, false)
            }
            LabelFilterOp::RegexEqual => {
                self.add(&filter.label, &filter.value, false, true)
            }
            LabelFilterOp::RegexNotEqual => {
                self.add(&filter.label, &filter.value, true, true)
            }
        }
    }

    pub fn is_match(&self, b: &str) -> bool {
        // todo: should sort first
        self.0.iter().all(|tf| tf.matches(b))
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

    pub fn match_cost(&self) -> usize {
        self.0.iter().map(|tf| tf.match_cost).sum()
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

// A translation of Matchers to an executable form.
pub struct LabelMatchers {
    pub filters: SmallVec<[TagFilters; 4]>,
}

impl Default for LabelMatchers {
    fn default() -> Self {
        Self {
            filters: SmallVec::new(),
        }
    }
}

impl LabelMatchers {

    pub fn from_matchers(matchers: &Matchers) -> RuntimeResult<Self> {
        let mut result = LabelMatchers::default();

        if !matchers.or_matchers.is_empty() {
            for label_filters in matchers.or_matchers.iter() {
                let mut filters = Self::filters_from_vec(&label_filters)?;
                filters.sort();
                result.filters.push(filters);
            }
        }

        if !matchers.matchers.is_empty() {
            let mut filters = Self::filters_from_vec(&matchers.matchers)?;
            filters.sort();
            result.filters.push(filters);
        }

        Ok(result)
    }

    pub fn is_match(&self, b: &str) -> bool {
        self.filters.iter().any(|m| m.is_match(b))
    }

    fn filters_from_vec(items: &[LabelFilter]) -> RuntimeResult<TagFilters> {
        let mut filters = TagFilters::new(Vec::with_capacity(items.len()));
        for filter in items.iter() {
            filters.add_label_filter(filter)?;
        }
        Ok(filters)
    }
}
