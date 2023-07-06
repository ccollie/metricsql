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
pub struct BinaryEvaluatorVectorScalar {
    op: Operator,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    keep_metric_names: bool,
    handler: BinopFunc,
}

impl BinaryEvaluatorVectorScalar {
    pub(crate) fn new(
        op: Operator,
        left: &Expr,
        right: &Expr,
        bool_modifier: bool,
        keep_metric_names: bool,
    ) -> RuntimeResult<Self> {
        debug_assert!(left.return_type() == ValueType::InstantVector);
        debug_assert!(right.return_type() == ValueType::Scalar);

        let lhs = Box::new(create_evaluator(left)?);
        let rhs = Box::new(create_evaluator(right)?);
        let handler = get_scalar_binop_handler(op, bool_modifier);

        Ok(Self {
            op,
            lhs,
            rhs,
            keep_metric_names,
            handler,
        })
    }
}

impl Value for BinaryEvaluatorVectorScalar {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Evaluator for BinaryEvaluatorVectorScalar {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        use QueryValue::*;

        let _ = if ctx.trace_enabled() {
            trace_span!(
                "vector scalar binary op",
                "op" = self.op.as_str(),
                series = field::Empty
            )
        } else {
            Span::none()
        }
        .entered();

        let right = self.rhs.eval(ctx, ec)?;
        let left = self.lhs.eval(ctx, ec)?;
        match (left, right) {
            (InstantVector(mut vector), Scalar(scalar)) => {
                // should not happen, but we can handle it
                for v in vector.iter_mut() {
                    if !self.keep_metric_names {
                        v.metric_name.reset_metric_group();
                    }

                    // todo: Rayon if over a threshold length
                    for value in v.values.iter_mut() {
                        *value = (self.handler)(*value, scalar);
                    }
                }
                Ok(InstantVector(std::mem::take(&mut vector)))
            }
            _ => {
                let msg = format!(
                    "Invalid argument type for binary operation: {} {} {}",
                    self.lhs.return_type(),
                    self.op,
                    self.rhs.return_type()
                );
                Err(RuntimeError::ArgumentError(msg))
            }
        }
    }

    fn volatility(&self) -> Volatility {
        Volatility::Stable
    }

    fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}
