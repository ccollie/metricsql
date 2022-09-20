use metricsql::ast::NumberExpr;
use metricsql::functions::Volatility;

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::eval_number;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct NumberEvaluator {
    value: f64
}

impl NumberEvaluator {
    pub fn new(expr: &NumberExpr) -> Self {
        Self {
            value: expr.value
        }
    }
}

impl Evaluator for NumberEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        Ok(eval_number(ec, self.value))
    }

    fn volatility(&self) -> Volatility {
        Volatility::Immutable
    }
}

impl From<f64> for NumberEvaluator {
    fn from(v: f64) -> Self {
        NumberEvaluator { value: v }
    }
}

impl From<i64> for NumberEvaluator {
    fn from(v: i64) -> Self {
        NumberEvaluator { value: v as f64 }
    }
}