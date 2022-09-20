use metricsql::ast::DurationExpr;
use metricsql::functions::Volatility;

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::eval_number;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct DurationEvaluator {
    expr: DurationExpr
}

impl DurationEvaluator {
    pub fn new(expr: &DurationExpr) -> Self {
        Self {
            expr: expr.clone()
        }
    }
}

impl Evaluator for DurationEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let d = self.expr.duration(ec.step);
        let d_sec: f64 = (d / 1000) as f64;
        Ok(eval_number(ec, d_sec))
    }

    fn volatility(&self) -> Volatility {
        if self.expr.requires_step {
            Volatility::Stable
        } else {
            Volatility::Immutable
        }
    }
}