use std::sync::Arc;
use metricsql::functions::{DataType, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::functions::types::AnyValue;
use crate::runtime_error::RuntimeResult;

/// An interface for evaluation of expressions
pub trait Evaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue>;

    fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }
    fn return_type(&self) -> DataType { DataType::InstantVector }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NullEvaluator {}

impl Evaluator for NullEvaluator {
    fn eval(&self, _ctx: &Arc<&Context>, _ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        Ok(AnyValue::Scalar(0.0))
    }
}
