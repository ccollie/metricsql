use metricsql::functions::Volatility;

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::eval_string;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct StringEvaluator {
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
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(eval_string(ec, &self.value))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }
}