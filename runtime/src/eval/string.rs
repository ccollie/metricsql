use metricsql::{StringExpr};
use crate::eval::{eval_string};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::eval::traits::Evaluator;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct StringEvaluator {
    value: String
}

impl StringEvaluator {
    pub fn new(expr: &StringExpr) -> Self {
        Self {
            value: expr.s.clone()
        }
    }
}

impl Evaluator for StringEvaluator {
    /// Evaluates and returns the result.
    fn eval<'a>(&self, ec: &'a mut EvalConfig) -> RuntimeResult<&'a Vec<Timeseries>> {
        Ok(&eval_string(ec, &self.value))
    }
}