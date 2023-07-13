use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql::ast::BinaryExpr;
use metricsql::binaryop::get_scalar_binop_handler;

use crate::{Context, InstantVector, QueryValue, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   http_requests_total{} * 2
///   http_requests_total{method="GET"} / 10
pub(crate) fn eval_vector_scalar_binop(
    ctx: &Arc<Context>,
    be: &BinaryExpr,
    vector: InstantVector,
    scalar: f64,
) -> RuntimeResult<QueryValue> {
    use QueryValue::*;

    let _ = if ctx.trace_enabled() {
        trace_span!(
            "vector scalar binary op",
            "op" = be.op.as_str(),
            series = field::Empty
        )
    } else {
        Span::none()
    }
    .entered();

    let handler = get_scalar_binop_handler(be.op, be.bool_modifier);

    let mut vector = vector;
    // should not happen, but we can handle it
    for v in vector.iter_mut() {
        if !be.keep_metric_names {
            v.metric_name.reset_metric_group();
        }

        for value in v.values.iter_mut() {
            *value = handler(*value, scalar);
        }
    }
    Ok(InstantVector(vector))
}
