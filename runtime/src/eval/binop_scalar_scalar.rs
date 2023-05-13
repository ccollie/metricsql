use std::sync::Arc;
use metricsql::ast::Expr;
use metricsql::binaryop::{BinopFunc, get_scalar_binop_handler};
use metricsql::common::{Operator, Value, ValueType};
use crate::eval::{create_evaluator, Evaluator, ExprEvaluator};
use crate::{Context, EvalConfig, QueryValue, RuntimeResult};

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
    pub(crate) fn new(op: Operator, left: &Expr, right: &Expr, is_bool: bool) -> RuntimeResult<Self> {
        debug_assert!(left.return_type() == ValueType::Scalar);
        debug_assert!(right.return_type() == ValueType::Scalar);
        let lhs = Box::new(create_evaluator(left)?);
        let rhs = Box::new(create_evaluator(right)?);
        let handler = get_scalar_binop_handler(op, is_bool);
        Ok(Self { op, lhs, rhs, handler })
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
            _ => unreachable!(), // todo: panic !
        }
    }

    fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}