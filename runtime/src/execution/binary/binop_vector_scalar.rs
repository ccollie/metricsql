use tracing::{field, trace_span, Span};

use metricsql::binaryop::get_scalar_binop_handler;
use metricsql::common::Operator;

use crate::{InstantVector, QueryValue, RuntimeResult};

/// eval_vector_scalar_binop evaluates binary operation between vector and scalar.
/// Ex:
///   http_requests_total{} * 2
///   http_requests_total{method="GET"} / 10
pub(crate) fn eval_vector_scalar_binop(
    vector: InstantVector,
    op: Operator,
    scalar: f64,
    bool_modifier: bool,
    _keep_metric_names: bool,
    is_tracing: bool,
) -> RuntimeResult<QueryValue> {
    use QueryValue::*;

    let _ = if is_tracing {
        trace_span!(
            "vector scalar binary op",
            "op" = op.as_str(),
            series = field::Empty
        )
    } else {
        Span::none()
    }
    .entered();

    let mut vector = vector;

    let is_unless = op == Operator::Unless;

    let handler = get_scalar_binop_handler(op, bool_modifier);
    for v in vector.iter_mut() {
        // reset_metric_group_if_required(be, v);

        // special case `unless` operator. If the vector has labels, then By definition we have mismatched
        // labels, since rhs is a scalar. In that case, return the vector as is.
        if is_unless && !v.metric_name.is_empty() {
            continue;
        }

        for value in v.values.iter_mut() {
            *value = handler(*value, scalar);
        }
    }

    Ok(InstantVector(vector))
}
