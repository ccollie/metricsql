use metricsql::ast::StringExpr;
use crate::eval::{eval_string};
use crate::runtime_error::{RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::context::Context;
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
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(eval_string(ec, &self.value))
    }
}