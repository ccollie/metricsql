use std::default::Default;

use metricsql::ast::DurationExpr;

use crate::execution::dag::subquery_node::SubqueryNode;
use crate::execution::dag::transform_node::AbsentTransformNode;
use crate::execution::dag::vector_vector_binary_node::VectorVectorPushDownNode;
use crate::execution::{Context, EvalConfig};
use crate::{QueryValue, RuntimeResult};

use super::aggregate_node::AggregateNode;
use super::binop_node::BinopNode;
use super::duration_node::DurationNode;
use super::dynamic_node::DynamicNode;
use super::rollup_node::RollupNode;
use super::scalar_vector_binop_node::ScalarVectorBinaryNode;
use super::selector_node::SelectorNode;
use super::transform_node::TransformNode;
use super::vector_scalar_binop_node::VectorScalarBinaryNode;
use super::vector_vector_binary_node::VectorVectorBinaryNode;

pub trait ExecutableNode {
    // separated as a method so we don't have to clone() args or run afoul of the borrow
    // checker with the dependencies
    fn set_dependencies(&mut self, _dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum DAGNode {
    Value(QueryValue),
    Aggregate(AggregateNode),
    Absent(AbsentTransformNode),
    Transform(TransformNode),
    Rollup(RollupNode),
    Duration(DurationNode),
    Selector(SelectorNode),
    Subquery(SubqueryNode),
    ScalarVectorOp(ScalarVectorBinaryNode),
    VectorVectorOp(VectorVectorBinaryNode),
    VectorVectorPushDownOp(VectorVectorPushDownNode),
    VectorScalarOp(VectorScalarBinaryNode),
    BinOp(BinopNode),
    Dynamic(DynamicNode),
}

impl DAGNode {
    pub fn is_value(&self) -> bool {
        matches!(self, DAGNode::Value(_))
    }

    pub fn execute(&mut self, context: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        ExecutableNode::execute(self, context, ec)
    }
}

impl ExecutableNode for DAGNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        match self {
            DAGNode::Value(_) => Ok(()),
            DAGNode::Absent(node) => node.set_dependencies(dependencies),
            DAGNode::Aggregate(node) => node.set_dependencies(dependencies),
            DAGNode::Transform(node) => node.set_dependencies(dependencies),
            DAGNode::Rollup(node) => node.set_dependencies(dependencies),
            DAGNode::Duration(node) => node.set_dependencies(dependencies),
            DAGNode::Selector(node) => node.set_dependencies(dependencies),
            DAGNode::ScalarVectorOp(node) => node.set_dependencies(dependencies),
            DAGNode::Subquery(sub) => sub.set_dependencies(dependencies),
            DAGNode::VectorVectorOp(node) => node.set_dependencies(dependencies),
            DAGNode::VectorScalarOp(node) => node.set_dependencies(dependencies),
            DAGNode::BinOp(node) => node.set_dependencies(dependencies),
            DAGNode::Dynamic(node) => node.set_dependencies(dependencies),
            DAGNode::VectorVectorPushDownOp(node) => node.set_dependencies(dependencies),
        }
    }
    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        match self {
            DAGNode::Value(node) => Ok(node.clone()),
            DAGNode::Aggregate(node) => node.execute(ctx, ec),
            DAGNode::Absent(node) => node.execute(ctx, ec),
            DAGNode::Transform(node) => node.execute(ctx, ec),
            DAGNode::Rollup(node) => node.execute(ctx, ec),
            DAGNode::Duration(node) => node.execute(ctx, ec),
            DAGNode::ScalarVectorOp(node) => node.execute(ctx, ec),
            DAGNode::Subquery(node) => node.execute(ctx, ec),
            DAGNode::VectorVectorOp(node) => node.execute(ctx, ec),
            DAGNode::VectorScalarOp(node) => node.execute(ctx, ec),
            DAGNode::BinOp(node) => node.execute(ctx, ec),
            DAGNode::Dynamic(node) => node.execute(ctx, ec),
            DAGNode::VectorVectorPushDownOp(node) => node.execute(ctx, ec),
            DAGNode::Selector(_) => {
                todo!("selector")
            }
        }
    }
}

impl Default for DAGNode {
    fn default() -> Self {
        DAGNode::Value(QueryValue::Scalar(f64::NAN))
    }
}

impl From<String> for DAGNode {
    fn from(s: String) -> Self {
        DAGNode::Value(QueryValue::String(s))
    }
}

impl From<&str> for DAGNode {
    fn from(s: &str) -> Self {
        DAGNode::Value(QueryValue::String(s.to_string()))
    }
}

impl From<f64> for DAGNode {
    fn from(f: f64) -> Self {
        DAGNode::Value(QueryValue::Scalar(f))
    }
}

impl From<i64> for DAGNode {
    fn from(i: i64) -> Self {
        DAGNode::Value(QueryValue::Scalar(i as f64))
    }
}

impl From<DurationExpr> for DAGNode {
    fn from(de: DurationExpr) -> Self {
        if !de.requires_step() {
            let val = de.value(1);
            let d_sec = val as f64 / 1000_f64;
            DAGNode::from(d_sec)
        } else {
            DAGNode::Duration(DurationNode(de))
        }
    }
}

impl From<&DurationExpr> for DAGNode {
    fn from(de: &DurationExpr) -> Self {
        if !de.requires_step() {
            let val = de.value(1);
            let d_sec = val as f64 / 1000_f64;
            DAGNode::from(d_sec)
        } else {
            DAGNode::Duration(DurationNode(de.clone()))
        }
    }
}
