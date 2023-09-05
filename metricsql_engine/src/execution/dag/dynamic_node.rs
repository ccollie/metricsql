use crate::execution::dag::dag_evaluator::DAGEvaluator;
use crate::execution::{Context, EvalConfig};
use crate::{QueryValue, RuntimeResult};

use super::ExecutableNode;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DynamicNode(pub DAGEvaluator);

impl ExecutableNode for DynamicNode {
    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        self.0.evaluate(ctx, ec)
    }
}
