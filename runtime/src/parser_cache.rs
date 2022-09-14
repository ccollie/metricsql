use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use lru_time_cache::{LruCache};

use metricsql::ast::{Expression};
use metricsql::optimizer::optimize;
use metricsql::parser::ParseError;
use crate::binary_op::adjust_cmp_ops;
use crate::create_evaluator;
use crate::eval::{ExprEvaluator, NullEvaluator};

const PARSE_CACHE_MAX_LEN: usize = 500;

pub struct ParseCacheValue {
    pub expr: Option<Expression>,
    pub evaluator: Option<ExprEvaluator>,
    pub err: Option<ParseError>
}

pub struct ParseCache {
    requests: AtomicU64,
    misses: AtomicU64,
    lru: Arc<LruCache<String, Arc<ParseCacheValue>>>, // todo: use parking_lot rwLock
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
            lru: Arc::new(LruCache::with_capacity(capacity)),
        }
    }

    pub fn len(&self) -> usize {
        self.lru.len()
    }

    pub fn misses(&self) -> u64 {
        self.misses.fetch_add(0, Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.fetch_add(0, Ordering::Relaxed)
    }

    pub fn clear(&mut self) {
        self.lru.clear()
    }

    pub fn parse(&mut self, q: &str) -> Arc<ParseCacheValue> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let entry = self.lru.entry(q.to_string()).or_insert_with(|| {
            self.misses.fetch_add(1, Ordering::Relaxed);
            let parsed = self.parse_internal(q);
            Arc::new(parsed)
        });
        entry.clone()
    }

    fn parse_internal(&mut self, q: &str) -> ParseCacheValue {
        match metricsql::parser::parse(q) {
            Ok(expr) => {
                let mut expression = &optimize(&expr);
                adjust_cmp_ops(&mut expression);
                match create_evaluator(expression) {
                    Ok(evaluator) => {
                        ParseCacheValue {
                            expr: Some(expr),
                            evaluator: Some(evaluator),
                            err: None,
                        }
                    },
                    Err(e) => {
                        ParseCacheValue {
                            expr: Some(expr),
                            evaluator: Some(ExprEvaluator::Null(NullEvaluator{})),
                            err: Some( ParseError::General("Error creating evaluator".to_string())),
                        }
                    }
                }
            },
            Err(e) => {
                ParseCacheValue {
                    expr: None,
                    evaluator: Some(ExprEvaluator::Null(NullEvaluator{})),
                    err: Some(e.clone()),
                }
            }
        }
    }
}
