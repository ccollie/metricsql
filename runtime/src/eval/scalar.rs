use std::sync::Arc;
use metricsql::ast::NumberExpr;
use metricsql::functions::{DataType, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::functions::types::AnyValue;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScalarEvaluator {
    value: f64
}

impl ScalarEvaluator {
    pub fn new(expr: &NumberExpr) -> Self {
        Self {
            value: expr.value
        }
    }
}

impl Evaluator for ScalarEvaluator {
    fn eval(&self, _ctx: &Arc<&Context>, _: &EvalConfig) -> RuntimeResult<AnyValue> {
        Ok(AnyValue::Scalar(self.value))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }

    fn return_type(&self) -> DataType {
        DataType::Scalar
    }
}

impl From<f64> for ScalarEvaluator {
    fn from(v: f64) -> Self {
        ScalarEvaluator { value: v }
    }
}

impl From<i64> for ScalarEvaluator {
    fn from(v: i64) -> Self {
        ScalarEvaluator { value: v as f64 }
    }
}