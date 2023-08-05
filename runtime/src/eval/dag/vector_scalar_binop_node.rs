use serde::{Deserialize, Serialize};

use metricsql::common::Operator;

use crate::eval::binary::eval_vector_scalar_binop;
use crate::{Context, EvalConfig, InstantVector, QueryValue, RuntimeResult};

use super::utils::resolve_vector;
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorScalarBinaryNode {
    pub right: f64,
    pub op: Operator,
    pub bool_modifier: bool,
    pub keep_metric_names: bool,
    pub left_idx: usize,
    #[serde(skip)]
    pub(crate) left: InstantVector,
}

impl ExecutableNode for VectorScalarBinaryNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.left = resolve_vector(self.left_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        eval_vector_scalar_binop(
            std::mem::take(&mut self.left),
            self.op,
            self.right,
            self.bool_modifier,
            self.keep_metric_names,
            ctx.trace_enabled(),
        )
    }
}
