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
    pub sort_results: bool,
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
                let optimized = optimize(expr);
                if let Ok(mut expression) = optimized {
                    adjust_comparison_ops(&mut expression);
                    let has_subquery = expression.contains_subquery();
                    let sort_results = should_sort_results(&expression);
                    ParseCacheValue {
                        expr: Some(expression),
                        err: None,
                        has_subquery,
                        sort_results,
                    }
                } else {
                    let err = optimized.err().unwrap();
                    ParseCacheValue {
                        expr: None,
                        has_subquery: false,
                        sort_results: false,
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
                sort_results: false,
            },
        }
    }
}

fn should_sort_results(e: &Expr) -> bool {
    return match e {
        Expr::Function(fe) => !fe.function.may_sort_results(),
        Expr::Aggregation(ae) => !ae.function.may_sort_results(),
        _ => true,
    };
}

pub(crate) fn escape_dots_in_regexp_label_filters(expr: &mut Expr) {
    match expr {
        Expr::MetricExpression(me) => {
            for lfs in me.label_filters.iter_mut() {
                if lfs.is_regexp() {
                    lfs.value = escape_dots(&lfs.value);
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

pub(crate) fn escape_dots(s: &str) -> String {
    // count the number of dots in the string
    let dots_count = s.chars().filter(|&c| c == '.').count();
    if dots_count <= 0 {
        return s.to_string();
    }
    let mut result = String::with_capacity(s.len() + 2 * dots_count);
    let mut prev_ch = '\0';
    let mut chars_iter = s.chars().peekable();
    let mut i = 0;
    while let Some(ch) = chars_iter.next() {
        if ch == '.' && (i == 0 || prev_ch != '\\') && {
            if let Some(next) = chars_iter.peek() {
                (*next != '*' && *next != '+' && *next != '?' && *next != '{')
            } else {
                true
            }
        } {
            // Escape a dot if the following conditions are met:
            // - if it isn't escaped already, i.e. if there is no `\` char before the dot.
            // - if there is no regexp modifiers such as '*', '?' or '{' after the dot.
            result.push('\\');
            result.push('.');
        } else {
            result.push(ch);
        }
        prev_ch = ch;
        i += 1;
    }
    return result;
}

#[cfg(test)]
mod tests {
    use metricsql::parser::parse;

    use crate::parser_cache::{escape_dots, escape_dots_in_regexp_label_filters};

    #[test]
    fn test_escape_dots() {
        fn f(input: &str, expected: &str) {
            let result = escape_dots(input);
            assert_eq!(
                expected, result,
                "unexpected result for escape_dots(\"{input}\"); got\n{result}\nwant\n{expected}",
            )
        }

        f("", "");
        f("a", "a");
        f("foobar", "foobar");
        f(".", r#"\."#);
        f(".*", ".*");
        f(".+", ".+");
        f("..", r#"\.\."#);
        f("foo.b.{2}ar..+baz.*", r#"foo\.b.{2}ar\..+baz.*"#)
    }

    #[test]
    fn test_escape_dots_in_regexp_label_filters() {
        fn f(s: &str, expected: &str) {
            let mut e = parse(s).unwrap();
            escape_dots_in_regexp_label_filters(&mut e);
            let result = e.to_string();
            assert_eq!(
                expected, result,
                "unexpected result for escape_dots_in_regexp_label_filters({s}); got\n{}\nwant\n{}",
                result, expected
            );
        }
        f("2", "2");
        f("foo.bar + 123", "foo.bar + 123");
        f(r#"avg{bar=~"baz.xx.yyy"}"#, r#"avg{bar=~"baz\\.xx\\.yyy"}"#);
        f(
            r#"avg(a.b{c="d.e",x=~"a.b.+[.a]",y!~"aaa.bb|cc.dd"}) + max(1,sum({x=~"aa.bb"}))"#,
            r#"avg(a.b{c="d.e",x=~"a\\.b.+[\\.a]",y!~"aaa\\.bb|cc\\.dd"}) + max(1, sum({x=~"aa\\.bb"}))"#,
        )
    }
}
