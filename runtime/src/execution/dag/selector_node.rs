use serde::{Deserialize, Serialize};

use metricsql::ast::MetricExpr;

use crate::execution::{Context, EvalConfig};
use crate::{QueryValue, RuntimeResult};

use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SelectorNode {
    pub metric: MetricExpr,
}

impl ExecutableNode for SelectorNode {
    fn set_dependencies(&mut self, _dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        Ok(())
    }

    fn execute(&mut self, _ctx: &Context, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        todo!("selector")
    }
}
