use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql::ast::Expr;
use metricsql::binaryop::{get_scalar_binop_handler, BinopFunc};
use metricsql::common::{Operator, Value, ValueType};
use metricsql::functions::Volatility;

use crate::eval::{create_evaluator, Evaluator, ExprEvaluator};
use crate::{Context, EvalConfig, QueryValue, RuntimeError, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   http_requests_total{} * 2
///   http_requests_total{method="GET"} / 10
pub(crate) fn eval_vector_scalar_binop(
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
            "vector scalar binary op",
            "op" = op.as_str(),
            series = field::Empty
        )
    } else {
        Span::none()
    }
    .entered();

    let handler = get_scalar_binop_handler(op, bool_modifier);

    // should not happen, but we can handle it
    for v in vector.iter_mut() {
        if !keep_metric_names {
            v.metric_name.reset_metric_group();
        }

        for value in v.values.iter_mut() {
            *value = handler(*value, scalar);
        }
    }
    Ok(InstantVector(std::mem::take(&mut vector)))
}
