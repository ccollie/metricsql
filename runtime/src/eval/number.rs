use metricsql::ast::NumberExpr;
use crate::runtime_error::{RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::{eval_number};
use crate::eval::traits::Evaluator;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct NumberEvaluator {
    value: f64
}

impl NumberEvaluator {
    pub fn new(expr: &NumberExpr) -> Self {
        Self {
            value: expr.n
        }
    }
}

impl Evaluator for NumberEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(eval_number(ec, self.value))
    }
}