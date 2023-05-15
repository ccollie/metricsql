use metricsql::common::ValueType;
use metricsql::functions::Volatility;
use metricsql::prelude::Value;
use std::sync::Arc;

use crate::context::Context;
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

/// An interface for evaluation of expressions
pub trait Evaluator: Value {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue>;

    fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }
    fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NullEvaluator {}

impl Value for NullEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Evaluator for NullEvaluator {
    fn eval(&self, _ctx: &Arc<Context>, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        Ok(QueryValue::Scalar(0.0))
    }
}
