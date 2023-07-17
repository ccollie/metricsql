use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lru_time_cache::LruCache;

use metricsql::ast::{adjust_comparison_ops, optimize, Expr};
use metricsql::parser;
use metricsql::parser::ParseError;

const PARSE_CACHE_MAX_LEN: usize = 500;

pub struct ParseCacheValue {
    pub expr: Option<Expr>,
    pub err: Option<ParseError>,
    pub has_subquery: bool,
}

pub struct ParseCache {
    requests: AtomicU64,
    misses: AtomicU64,
    lru: Mutex<LruCache<String, Arc<ParseCacheValue>>>, // todo: use parking_lot rwLock
}

#[derive(PartialEq)]
pub enum ParseCacheResult {
    CacheHit,
    CacheMiss,
}

impl Default for ParseCache {
    fn default() -> Self {
        ParseCache::new(PARSE_CACHE_MAX_LEN)
    }
}

impl ParseCache {
    pub fn new(capacity: usize) -> Self {
        ParseCache {
            requests: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            lru: Mutex::new(LruCache::with_capacity(capacity)),
        }
    }

    pub fn len(&self) -> usize {
        self.lru.lock().unwrap().len()
    }

    pub fn misses(&self) -> u64 {
        self.misses.fetch_add(0, Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.fetch_add(0, Ordering::Relaxed)
    }

    pub fn clear(&mut self) {
        self.lru.lock().unwrap().clear()
    }

    pub fn parse(&self, q: &str) -> (Arc<ParseCacheValue>, ParseCacheResult) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        match self.get(q) {
            Some(value) => (value, ParseCacheResult::CacheHit),
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                let parsed = Self::parse_internal(q);
                let k = q.to_string();
                let to_insert = Arc::new(parsed);

                let mut lru = self.lru.lock().unwrap();
                lru.insert(k, to_insert.clone());
                (to_insert, ParseCacheResult::CacheMiss)
            }
        }
    }

    pub fn get(&self, q: &str) -> Option<Arc<ParseCacheValue>> {
        // use interior mutability here
        let mut lru = self.lru.lock().unwrap();
        match lru.get(q) {
            None => None,
            Some(v) => Some(Arc::clone(v)),
        }
    }

    // todo: pass options
    fn parse_internal(q: &str) -> ParseCacheValue {
        match parser::parse(q) {
            Ok(expr) => {
                let mut optimized = optimize(expr);
                if let Ok(mut expression) = optimized {
                    adjust_comparison_ops(&mut expression);
                    let has_subquery = expression.contains_subquery();
                    ParseCacheValue {
                        expr: Some(expression),
                        err: None,
                        has_subquery,
                    }
                } else {
                    let err = optimized.err().unwrap();
                    ParseCacheValue {
                        expr: None,
                        has_subquery: false,
                        err: Some(ParseError::General(format!(
                            "Error optimizing expression: {:?}",
                            err
                        ))),
                    }
                }
            }
            Err(e) => ParseCacheValue {
                expr: None,
                err: Some(e.clone()),
                has_subquery: false,
            },
        }
    }
}

fn escape_dots_in_regexp_label_filters(expr: &mut Expr) {
    match expr {
        Expr::MetricExpression(me) => {
            for lfs in me.label_filterss.iter_mut() {
                for f in lfs.iter_mut() {
                    if f.is_regexp {
                        f.value = escape_dots_in_regexp_label_filters(f.value.as_ref());
                    }
                }
            }
        }
        Expr::Aggregation(agg) => {
            for arg in agg.args.iter_mut() {
                escape_dots_in_regexp_label_filters(arg);
            }
        }
        Expr::BinaryOperator(be) => {
            escape_dots_in_regexp_label_filters(&mut be.left);
            escape_dots_in_regexp_label_filters(&mut be.right);
        }
        Expr::Function(fe) => {
            for arg in fe.args.iter_mut() {
                escape_dots_in_regexp_label_filters(arg);
            }
        }
        Expr::Parens(pe) => {
            for e in pe.expressions.iter_mut() {
                escape_dots_in_regexp_label_filters(e);
            }
        }
        Expr::Rollup(re) => {
            escape_dots_in_regexp_label_filters(&mut re.expr);
        }
        _ => {}
    }
}

fn escape_dots(s: &str) -> String {
    let dots_count = strings.Count(s, ".");
    if dots_count <= 0 {
        return s.to_string();
    }
    let mut result = String::with_capacity(s.len() + 2 * dots_count);
    let len = s.len();
    let mut prev_ch = '\0';
    for (i, ch) in s.chars().enumerate() {
        if ch == '.'
            && (i == 0 || prev_ch != '\\')
            && (i + 1 == len
                || i + 1 < len && s[i + 1] != '*' && s[i + 1] != '+' && s[i + 1] != '{')
        {
            // Escape a dot if the following conditions are met:
            // - if it isn't escaped already, i.e. if there is no `\` char before the dot.
            // - if there is no regexp modifiers such as '+', '*' or '{' after the dot.
            result.push('\\');
            result.push('.');
        } else {
            result.push(ch);
        }
        prev_ch = ch;
    }
    return result;
}
