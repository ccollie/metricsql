use metricsql::{DurationExpr};
use crate::eval::{eval_number};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::eval::traits::Evaluator;
use crate::rollup::{get_rollup_func, NewRollupFunc};
use crate::traits::Evaluator;

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct DurationEvaluator {
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
    fn eval(&self, ec: &mut EvalConfig) -> RuntimeResult<&Vec<Timeseries>> {
        let d = self.expr.duration(ec.step)?;
        let d_sec: f64 = (d / 1000) as f64;
        Ok(&eval_number(ec, d_sec))
    }
}