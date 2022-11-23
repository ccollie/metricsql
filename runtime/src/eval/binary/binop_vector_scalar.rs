use tracing::{field, trace_span, Span};

use metricsql::ast::BinaryExpr;
use metricsql::binaryop::get_scalar_binop_handler;

use crate::{InstantVector, QueryValue, RuntimeResult};

use super::reset_metric_group_if_required;

/// eval_vector_scalar_binop evaluates binary operation between vector and scalar.
/// Ex:
///   http_requests_total{} * 2
///   http_requests_total{method="GET"} / 10
pub(crate) fn eval_vector_scalar_binop(
    be: &BinaryExpr,
    vector: InstantVector,
    scalar: f64,
    is_tracing: bool,
) -> RuntimeResult<QueryValue> {
    use QueryValue::*;

    let _ = if is_tracing {
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

    for v in vector.iter_mut() {
        reset_metric_group_if_required(be, v);

        for value in v.values.iter_mut() {
            *value = handler(*value, scalar);
        }
    }

    Ok(InstantVector(vector))
}
