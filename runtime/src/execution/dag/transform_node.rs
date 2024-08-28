use tracing::{field, trace_span, Span};

use metricsql_parser::prelude::TransformFunction;

use crate::execution::dag::utils::resolve_node_args;
use crate::execution::{Context, EvalConfig};
use crate::functions::transform::{exec_transform_fn, TransformFuncArg};
use crate::{QueryValue, RuntimeResult};

use super::{ExecutableNode, NodeArg};

#[derive(Debug, Clone, PartialEq)]
pub struct TransformNode {
    pub function: TransformFunction,
    pub keep_metric_names: bool,
    pub node_args: Vec<NodeArg>,
    pub(crate) args: Vec<QueryValue>,
    /// Whether all arguments are constant and fully resolved. Note that if this is true,
    /// this node will be either be executed standalone or as the first level of a DAG evaluator.
    /// `set_dependencies` will therefore not be called.
    pub(crate) args_const: bool,
}

impl ExecutableNode for TransformNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        resolve_node_args(&self.node_args, &mut self.args, dependencies);
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let is_tracing = ctx.trace_enabled();

        let span = if is_tracing {
            trace_span!(
                "transform",
                function = self.function.name(),
                series = field::Empty
            )
        } else {
            Span::none()
        }
        .entered();

        let mut tfa = TransformFuncArg {
            args: if self.args_const {
                self.args.clone()
            } else {
                std::mem::take(&mut self.args)
            },
            ec,
            keep_metric_names: self.keep_metric_names,
        };

        exec_transform_fn(self.function, &mut tfa).map(|res| {
            if is_tracing {
                span.record("series", res.len());
            }
            QueryValue::InstantVector(res)
        })
    }
}

impl Default for TransformNode {
    fn default() -> Self {
        Self {
            function: TransformFunction::Absent, // todo:
            args: vec![],
            keep_metric_names: false,
            args_const: false,
            node_args: vec![],
        }
    }
}
