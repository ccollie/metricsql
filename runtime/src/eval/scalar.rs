use metricsql::ast::NumberExpr;
use metricsql::functions::{Volatility};
use std::sync::Arc;
use metricsql::common::Value;
use metricsql::prelude::ValueType;

use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScalarEvaluator(f64);

impl ScalarEvaluator {
    pub fn new(expr: &NumberExpr) -> Self {
        Self(expr.value)
    }
}

impl Value for ScalarEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Evaluator for ScalarEvaluator {
    fn eval(&self, _ctx: &Arc<Context>, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        Ok(QueryValue::Scalar(self.0))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }

    fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl From<f64> for ScalarEvaluator {
    fn from(v: f64) -> Self {
        ScalarEvaluator(v)
    }
}

impl From<i64> for ScalarEvaluator {
    fn from(v: i64) -> Self {
        ScalarEvaluator(v as f64)
    }
}
