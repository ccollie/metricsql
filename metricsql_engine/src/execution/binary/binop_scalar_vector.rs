use tracing::{field, trace_span, Span};

use metricsql_parser::prelude::{get_scalar_binop_handler, Operator};

use crate::{InstantVector, QueryValue, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   2 * http_requests_total{}
///   42 - http_requests_total{method="GET"}
pub(crate) fn eval_scalar_vector_binop(
    scalar: f64,
    op: Operator,
    vector: InstantVector,
    bool_modifier: bool,
    keep_metric_names: bool,
    is_tracing: bool,
) -> RuntimeResult<QueryValue> {
    let _ = if is_tracing {
        trace_span!(
            "scalar vector binary op",
            "op" = op.as_str(),
            series = field::Empty
        )
    } else {
        Span::none()
    }
    .entered();

    let handler = get_scalar_binop_handler(op, bool_modifier);

    let mut vector = vector;

    for ts in vector.iter_mut() {
        if !keep_metric_names {
            ts.metric_name.reset_metric_group();
        }

        for value in ts.values.iter_mut() {
            *value = handler(scalar, *value);
        }
    }

    Ok(QueryValue::InstantVector(vector))
}
