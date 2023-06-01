use crate::eval::{create_evaluator, Evaluator, ExprEvaluator};
use crate::{Context, EvalConfig, QueryValue, RuntimeResult};
use metricsql::ast::Expr;
use metricsql::binaryop::{get_scalar_binop_handler, BinopFunc};
use metricsql::common::{Operator, Value, ValueType};
use std::sync::Arc;

/// BinaryEvaluatorScalarScalar
/// Ex:
///   1 + 2
///   42 / 6
///   2 ^ 64
///   ...
pub struct BinaryEvaluatorScalarScalar {
    op: Operator,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    handler: BinopFunc,
}

impl BinaryEvaluatorScalarScalar {
    pub(crate) fn new(
        op: Operator,
        left: &Expr,
        right: &Expr,
        is_bool: bool,
    ) -> RuntimeResult<Self> {
        debug_assert!(left.return_type() == ValueType::Scalar);
        debug_assert!(right.return_type() == ValueType::Scalar);
        let lhs = Box::new(create_evaluator(left)?);
        let rhs = Box::new(create_evaluator(right)?);
        let handler = get_scalar_binop_handler(op, is_bool);
        Ok(Self {
            op,
            lhs,
            rhs,
            handler,
        })
    }
}

impl Value for BinaryEvaluatorScalarScalar {
    fn value_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Evaluator for BinaryEvaluatorScalarScalar {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        use QueryValue::*;

        match (self.lhs.eval(ctx, ec)?, self.rhs.eval(ctx, ec)?) {
            (Scalar(left), Scalar(right)) => {
                let res = (self.handler)(left, right);
                Ok(Scalar(res))
            }
            (lhs, rhs) => {
                let msg = format!(
                    "expected scalar args for binary operator {:?}; got {:?} and {:?}",
                    self.op, lhs, rhs
                );
                unreachable!("{}", msg)
            }
        }
    }

    fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}
