use std::sync::Arc;

use tracing::{trace_span, Span};

use metricsql::binaryop::get_scalar_binop_handler;
use metricsql::common::{Operator, Value, ValueType};

use crate::{Context, QueryValue, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   2 * http_requests_total{}
///   42 - http_requests_total{method="GET"}
pub(crate) fn eval_scalar_vector_binop(
    ctx: &Arc<Context>,
    vector: InstantVector,
    scalar: f64,
    op: Operator,
    keep_metric_names: bool,
    bool_modifier: bool,
) -> RuntimeResult<QueryValue> {
    use QueryValue::*;

    let _ = if ctx.trace_enabled() {
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

    for v in vector.iter_mut() {
        if !keep_metric_names {
            v.metric_name.reset_metric_group();
        }
        for value in v.values.iter_mut() {
            *value = handler(scalar, *value);
        }
    }

    Ok(InstantVector(vector))
}
