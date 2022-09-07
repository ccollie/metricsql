use metricsql::ast::{DurationExpr, Expression};
use crate::runtime_error::RuntimeResult;

/// isRollup verifies whether s is a rollup with non-empty window.
///
/// It returns the wrapped query with the corresponding window, step and offset.
pub fn is_rollup(s: &str) -> RuntimeResult<Option<(String, DurationExpr, DurationExpr, DurationExpr)>> {
    let expr = parsePromQLWithCache(s)?;
    match expr {
        Expression::Rollup(r) => {
            let wrappedQuery = r.expr.to_string();
            Ok((wrappedQuery, r.window, r.step, r.offset))
        },
        _ => Ok(None)
    }
}
