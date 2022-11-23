use tracing::{field, trace_span, Span};

use metricsql::ast::BinaryExpr;
use metricsql::binaryop::get_scalar_binop_handler;

use crate::{InstantVector, QueryValue, RuntimeResult};

use super::reset_metric_group_if_required;

/// BinaryEvaluatorScalarVector
/// Ex:
///   2 * http_requests_total{}
///   42 - http_requests_total{method="GET"}
pub(crate) fn eval_scalar_vector_binop(
    be: &BinaryExpr,
    vector: InstantVector,
    scalar: f64,
    is_tracing: bool,
) -> RuntimeResult<QueryValue> {
    let _ = if is_tracing {
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
        reset_metric_group_if_required(be, ts);

        for value in ts.values.iter_mut() {
            *value = handler(scalar, *value);
        }
    }

    Ok(QueryValue::InstantVector(vector))
}
