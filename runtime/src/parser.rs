use std::ops::Deref;
use metricsql::ast::{DurationExpr, Expression};
use crate::context::Context;
use crate::parse_promql_with_cache;
use crate::runtime_error::RuntimeResult;

/// isRollup verifies whether s is a rollup with non-empty window.
///
/// It returns the wrapped query with the corresponding window, step and offset.
pub fn is_rollup(ctx: &mut Context, s: &str) -> RuntimeResult<Option<(String, DurationExpr, DurationExpr, DurationExpr)>> {
    let cached = parse_promql_with_cache(ctx,s)?;
    match &cached.expr {
        Some(Expression::Rollup(r)) => {
            let wrapped_query = r.expr.to_string();
            if r.window.is_none() {
                return Ok(None)
            }
            let window = r.window.unwrap().clone();
            Ok(Some(
                (
                    wrapped_query,
                    window,
                    get_duration_expr(&r.step),
                    get_duration_expr(&r.offset),
                )
            ))
        },
        _ => Ok(None)
    }
}

/// IsMetricSelectorWithRollup verifies whether s contains PromQL metric selector
/// wrapped into rollup.
///
/// It returns the wrapped query with the corresponding window with offset.
pub fn is_metric_selector_with_rollup(ctx: &mut Context, s: &str) -> RuntimeResult<Option<(String, DurationExpr, DurationExpr)>> {
    let cached = parse_promql_with_cache(ctx, s)?;
    match &cached.expr {
        Some(Expression::Rollup(r)) => {
            if r.window.is_none() || r.step.is_none() {
                return Ok(None);
            }
            match &r.expr.deref() {
                Expression::MetricExpression(me) => {
                    if me.label_filters.len() == 0 {
                        return Ok(None)
                    }
                },
                _ => return Ok(None)
            }
            let window = r.window.unwrap().clone();
            let wrapped_query = r.expr.to_string();

            let offset = get_duration_expr(&r.offset);

            Ok(Some((wrapped_query, window, offset)))
        },
        _ => Ok(None)
    }
}

fn get_duration_expr(offset: &Option<DurationExpr>) -> DurationExpr {
    return match &offset {
        Some(ofs) => ofs.clone(),
        None => DurationExpr::default()
    }
}