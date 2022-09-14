use crate::{EvalConfig, Timeseries};
use crate::runtime_error::RuntimeResult;
use clone_dyn::clone_dyn;
use crate::context::Context;
use crate::functions::types::Volatility;

/// An interface for evaluation of expressions
#[clone_dyn]
pub trait Evaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>>;

    fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct NullEvaluator {}

impl Evaluator for NullEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(vec![])
    }
}
