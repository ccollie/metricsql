use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use dashmap::mapref::one::RefMut;

use metricsql::ast::{DurationExpr, Expression};
use metricsql::optimizer::optimize;
use metricsql::parser::ParseError;
use crate::binary_op::adjust_cmp_ops;
use crate::create_evaluator;
use crate::eval::{ExprEvaluator, NullEvaluator};

const PARSE_CACHE_MAX_LEN: usize = 1000;

pub struct ParseCacheValue {
    pub expr: Option<Expression>,
    pub evaluator: Option<ExprEvaluator>,
    pub err: Option<ParseError>,
    pub(crate) ref_count: u64
}

pub struct ParseCache {
    requests: AtomicU64,
    misses: AtomicU64,
    hash: Arc<DashMap<String, ParseCacheValue>>,
}

impl ParseCache {
    pub fn new() -> Self {
        ParseCache {
            requests: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hash: Arc::new(DashMap::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.hash.len()
    }

    pub fn misses(&self) -> u64 {
        self.misses.fetch_add(0, Ordering::Relaxed)
    }

    pub fn requests(&self) -> u64 {
        self.requests.fetch_add(0, Ordering::Relaxed)
    }

    pub fn clear(&mut self) {
        self.hash.clear()
    }

    fn cleanup(&mut self) {
        let mut overflow = self.len() - PARSE_CACHE_MAX_LEN;
        if overflow > 0 {
            // Remove 10% of items from the cache.
            overflow = ((self.len() as f64) * 0.1) as usize;
            let mut items = self.hash.iter().collect();

            for k in self.hash.iter() {
                inner.m.remove(k);
                overflow -= 1;
                if overflow <= 0 {
                    break
                }
            }
        }
    }

    pub fn parse<'a>(&mut self, q: &str) -> RefMut<'a, String, ParseCacheValue> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let entry = self.hash.entry(q.to_string()).or_insert_with(|| {
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.parse_internal(q)
        });
        entry.ref_count += 1;
        entry
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
                            ref_count: 0
                        }
                    },
                    Err(e) => {
                        ParseCacheValue {
                            expr: Some(expr),
                            evaluator: Some(ExprEvaluator::Null(NullEvaluator{})),
                            err: Some( ParseError::General("Error creating evaluator".to_string())),
                            ref_count: 0
                        }
                    }
                }
            },
            Err(e) => {
                ParseCacheValue {
                    expr: None,
                    evaluator: Some(ExprEvaluator::Null(NullEvaluator{})),
                    err: Some(e.clone()),
                    ref_count: 0
                }
            }
        }
    }
}

/// IsMetricSelectorWithRollup verifies whether s contains PromQL metric selector
/// wrapped into rollup.
///
/// It returns the wrapped query with the corresponding window with offset.
pub fn is_metric_selector_with_rollup(s: &str) -> (String, DurationExpr, DurationExpr) {
    let expr = parsePromQLWithCache(s)?;
    match expr {
        Expression::Rollup(r) => {
            if r.window.is_none() || r.step.is_none() {
                return None;
            }
            match r.expr {
                Expression::MetricExpression(me) => {
                    if me.label_filters.len() == 0 {
                        return None
                    }
                }
            }
            let wrapped_query = r.expr.to_string();
            Ok((wrapped_query, r.window, r.offset))
        },
        _ => Ok(None)
    }
}
