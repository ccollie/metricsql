use std::sync::Arc;
use metricsql::ast::Expr;
use metricsql::binaryop::{BinopFunc, get_scalar_binop_handler};
use metricsql::common::{Operator, Value, ValueType};
use metricsql::functions::Volatility;
use crate::eval::{create_evaluator, Evaluator, ExprEvaluator};
use crate::{Context, EvalConfig, QueryValue, RuntimeError, RuntimeResult};

/// BinaryEvaluatorScalarVector
/// Ex:
///   2 * http_requests_total{}
///   42 - http_requests_total{method="GET"}
pub struct BinaryEvaluatorScalarVector {
    op: Operator,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    keep_metric_names: bool,
    handler: BinopFunc
}

impl BinaryEvaluatorScalarVector {
    pub(crate) fn new(
        op: Operator,
        left: &Expr,
        right: &Expr,
        bool_modifier: bool,
        keep_metric_names: bool
    ) -> RuntimeResult<Self> {
        debug_assert!(right.return_type() == ValueType::InstantVector);

        let lhs = Box::new(create_evaluator(left)?);
        let rhs = Box::new(create_evaluator(right)?);
        let handler = get_scalar_binop_handler(op, bool_modifier);

        Ok(Self {
            op,
            lhs,
            rhs,
            keep_metric_names,
            handler
        })
    }
}

impl Value for BinaryEvaluatorScalarVector {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Evaluator for BinaryEvaluatorScalarVector {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        use QueryValue::*;

        let right = self.rhs.eval(ctx, ec)?;
        let left = self.lhs.eval(ctx, ec)?;
        match (left, right) {
            (Scalar(scalar), InstantVector(mut vector)) => {
                for v in vector.iter_mut() {
                    if !self.keep_metric_names {
                        v.metric_name.reset_metric_group();
                    }
                    for value in v.values.iter_mut() {
                        *value = (self.handler)(scalar, *value);
                    }
                }
                Ok(InstantVector(std::mem::take(&mut vector)))
            }
            _=> {
                let msg = format!("Invalid argument type for binary operation: {} {} {}",
                                  self.lhs.return_type(), self.op, self.rhs.return_type());
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