use serde::{Deserialize, Serialize};

use metricsql_parser::ast::Operator;

use crate::execution::binary::eval_scalar_vector_binop;
use crate::execution::{Context, EvalConfig};
use crate::{InstantVector, QueryValue, RuntimeResult};

use super::utils::resolve_vector;
use super::ExecutableNode;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScalarVectorBinaryNode {
    pub left: f64,
    pub op: Operator,
    pub bool_modifier: bool,
    pub keep_metric_names: bool,
    pub right_idx: usize,
    #[serde(skip)]
    pub(crate) right: InstantVector,
}

impl ExecutableNode for ScalarVectorBinaryNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.right = resolve_vector(self.right_idx, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, _ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        eval_scalar_vector_binop(
            self.left,
            self.op,
            std::mem::take(&mut self.right),
            self.bool_modifier,
            self.keep_metric_names,
            ctx.trace_enabled(),
        )
    }
}
