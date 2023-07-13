use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql::ast::BinaryExpr;
use metricsql::binaryop::get_scalar_binop_handler;

use crate::{Context, InstantVector, QueryValue, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   2 * http_requests_total{}
///   42 - http_requests_total{method="GET"}
pub(crate) fn eval_scalar_vector_binop(
    ctx: &Arc<Context>,
    be: &BinaryExpr,
    mut vector: InstantVector,
    scalar: f64,
) -> RuntimeResult<QueryValue> {
    let _ = if ctx.trace_enabled() {
        trace_span!(
            "scalar vector binary op",
            "op" = be.op.as_str(),
            series = field::Empty
        )
    } else {
        Span::none()
    }
    .entered();

    let handler = get_scalar_binop_handler(be.op, be.bool_modifier);

    let mut vector = vector;

    for ts in vector.iter_mut() {
        if !be.keep_metric_names {
            ts.metric_name.reset_metric_group();
        }

        for value in ts.values.iter_mut() {
            *value = handler(scalar, *value);
        }
    }

    Ok(QueryValue::InstantVector(vector))
}
