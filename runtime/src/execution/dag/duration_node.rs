use metricsql_parser::ast::DurationExpr;
use serde::{Deserialize, Serialize};

use crate::execution::{Context, EvalConfig};
use crate::{QueryValue, RuntimeResult};

use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DurationNode(pub DurationExpr);

impl ExecutableNode for DurationNode {
    fn execute(&mut self, _ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let value = self.0.value(ec.step);
        Ok(QueryValue::Scalar(value as f64))
    }
}
