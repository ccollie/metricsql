use metricsql::{Expression, NumberExpr};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::eval::{eval_number};
use crate::eval::traits::Evaluator;
use crate::traits::Evaluator;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct NumberEvaluator {
    n: f64
}

impl NumberEvaluator {
    pub fn new(expr: &NumberExpr) -> Self {
        Self {
            n: expr.n
        }
    }
}

impl Evaluator for NumberEvaluator {
    /// Evaluates and returns the result.
    fn eval<'a>(&self, ec: &'a mut EvalConfig) -> RuntimeResult<&'a Vec<Timeseries>> {
        Ok(&eval_number(ec, self.n))
    }
}