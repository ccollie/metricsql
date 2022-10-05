use std::sync::Arc;
use metricsql::ast::DurationExpr;
use metricsql::functions::{DataType, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::traits::Evaluator;
use crate::functions::types::AnyValue;
use crate::runtime_error::RuntimeResult;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DurationEvaluator {
    expr: DurationExpr
}

impl DurationEvaluator {
    pub fn new(expr: &DurationExpr) -> Self {
        Self {
            expr: expr.clone()
        }
    }
    
    pub(super) fn is_const(&self) -> bool {
        !self.expr.requires_step
    }
}

impl Evaluator for DurationEvaluator {
    fn eval(&self, _ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        let d = self.expr.duration(ec.step);
        let d_sec: f64 = (d / 1000) as f64;
        Ok(AnyValue::Scalar(d_sec))
    }

    fn volatility(&self) -> Volatility {
        if self.expr.requires_step {
            Volatility::Stable
        } else {
            Volatility::Immutable
        }
    }

    fn return_type(&self) -> DataType {
        DataType::Scalar
    }
}