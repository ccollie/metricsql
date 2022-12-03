use metricsql::functions::{Volatility};
use std::sync::Arc;
use metricsql::common::{Value, ValueType};

use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringEvaluator(String);

impl StringEvaluator {
    pub fn new<S: Into<String>>(expr: S) -> Self {
        Self(expr.into())
    }
}

impl From<&str> for StringEvaluator {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

impl Value for StringEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::String
    }
}

impl Evaluator for StringEvaluator {
    /// Evaluates and returns the result.
    fn eval(&self, _ctx: &Arc<Context>, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        // todo: how to avoid clone ?
        Ok(QueryValue::String(self.0.clone()))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }

    fn return_type(&self) -> ValueType {
        ValueType::String
    }
}
