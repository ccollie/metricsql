use metricsql_parser::ast::MetricExpr;
use serde::{Deserialize, Serialize};

use crate::execution::{Context, EvalConfig};
use crate::{RuntimeResult};
use crate::types::QueryValue;
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SelectorNode {
    pub metric: MetricExpr,
}

impl ExecutableNode for SelectorNode {
    fn pre_execute(&mut self, _dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        Ok(())
    }

    fn execute(&mut self, _ctx: &Context, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        todo!("selector")
    }
}
