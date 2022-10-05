use std::sync::Arc;
use metricsql::functions::{DataType, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::functions::types::AnyValue;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringEvaluator {
    value: String
}

impl StringEvaluator {
    pub fn new(expr: &str) -> Self {
        Self {
            value: expr.to_string()
        }
    }
}

impl From<&str> for StringEvaluator {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

impl Evaluator for StringEvaluator {
    /// Evaluates and returns the result.
    fn eval(&self, _ctx: &Arc<&Context>, _ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        // todo: how to avoid clone ?
        Ok(AnyValue::String(self.value.clone()))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }

    fn return_type(&self) -> DataType {
        DataType::String
    }
}