pub use dag_node::*;
use metricsql_parser::prelude::Expr;

use crate::execution::dag::builder::DAGBuilder;
use crate::RuntimeResult;

mod aggregate_node;
mod binop_node;
pub(super) mod builder;
mod dag_evaluator;
mod dag_node;
mod duration_node;
mod dynamic_node;
mod rollup_node;
mod scalar_vector_binop_node;
mod selector_node;
mod subquery_node;
#[cfg(test)]
mod test_exec;
mod transform_node;
mod utils;
mod vector_scalar_binop_node;
mod vector_vector_binary_node;

pub fn compile_expression(expr: &Expr) -> RuntimeResult<DAGNode> {
    DAGBuilder::compile(expr.clone())
}
