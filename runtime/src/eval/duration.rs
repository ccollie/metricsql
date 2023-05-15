use metricsql::ast::DurationExpr;
use metricsql::common::{Value, ValueType};
use metricsql::functions::Volatility;
use std::sync::Arc;

use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DurationEvaluator(DurationExpr);

impl DurationEvaluator {
    pub fn new(expr: &DurationExpr) -> Self {
        Self(expr.clone())
    }

    pub(super) fn is_const(&self) -> bool {
        !self.0.requires_step
    }
}

impl Evaluator for DurationEvaluator {
    fn eval(&self, _ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let d = self.0.value(ec.step);
        let d_sec: f64 = (d / 1000) as f64;
        Ok(QueryValue::Scalar(d_sec))
    }

    fn volatility(&self) -> Volatility {
        if self.0.requires_step {
            Volatility::Stable
        } else {
            Volatility::Immutable
        }
    }

    fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Value for DurationEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::Scalar
    }
}
