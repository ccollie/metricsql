use lru_time_cache::LruCache;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use metricsql::ast::Expr;
use metricsql::optimize;
use metricsql::parser::ParseError;

use crate::eval::{create_evaluator, ExprEvaluator};

const PARSE_CACHE_MAX_LEN: usize = 500;

pub struct ParseCacheValue {
    pub expr: Option<Expr>,
    pub evaluator: Option<ExprEvaluator>,
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

    fn parse_internal(q: &str) -> ParseCacheValue {
        match metricsql::parser::parse(q) {
            Ok(expr) => {
                let optimized = optimize::optimize(expr);
                if let Ok(expression) = optimized {
                    match create_evaluator(&expression) {
                        Ok(evaluator) => {
                            let has_subquery = expression.contains_subquery();
                            ParseCacheValue {
                                expr: Some(expression),
                                evaluator: Some(evaluator),
                                err: None,
                                has_subquery,
                            }
                        }
                        Err(e) => ParseCacheValue {
                            expr: Some(expression),
                            evaluator: Some(ExprEvaluator::default()),
                            has_subquery: false,
                            err: Some(ParseError::General(format!(
                                "Error creating evaluator: {:?}",
                                e
                            ))),
                        },
                    }
                } else {
                    let err = optimized.err().unwrap();
                    ParseCacheValue {
                        expr: None,
                        evaluator: Some(ExprEvaluator::default()),
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
                evaluator: Some(ExprEvaluator::default()),
                err: Some(e.clone()),
                has_subquery: false,
            },
        }
    }
}
