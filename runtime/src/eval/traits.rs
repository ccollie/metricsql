use clone_dyn::clone_dyn;

use metricsql::functions::{DataType, Volatility};

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::runtime_error::RuntimeResult;

/// An interface for evaluation of expressions
#[clone_dyn]
pub trait Evaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>>;

    fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }
    fn return_type(&self) -> DataType { DataType::Series }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct NullEvaluator {}

impl Evaluator for NullEvaluator {
    fn eval(&self, _ctx: &mut Context, _ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(vec![])
    }
}
