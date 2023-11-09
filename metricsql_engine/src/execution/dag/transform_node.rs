use metricsql_parser::prelude::{Expr, TransformFunction};
use tracing::{field, trace_span, Span};

use crate::execution::dag::utils::resolve_node_args;
use crate::execution::{eval_number, Context, EvalConfig};
use crate::functions::transform::{
    exec_transform_fn, extract_labels, handle_absent, TransformFuncArg,
};
use crate::{Labels, QueryValue, RuntimeResult, Timeseries};

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

/// Hack. This is a copy of TransformNode. This exists solely because
/// `absent` is the only transform function that requires access to the
/// original expression in order to get the set of labels to apply to the result vector.
/// Rather than pass around a param that's redundant in 99.9% of case we handle it here and
/// hopefully keep the code cleaner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AbsentTransformNode {
    pub keep_metric_names: bool,
    pub labels: Option<Labels>,
    pub node_arg: NodeArg,
    pub(crate) arg: QueryValue,
    pub(crate) arg_const: bool,
}

impl AbsentTransformNode {
    pub fn from_arg_index(expr: &Expr, arg_index: usize) -> Self {
        let labels = extract_labels(expr);
        Self {
            keep_metric_names: false,
            labels,
            node_arg: NodeArg::Index(arg_index),
            arg: QueryValue::default(),
            arg_const: false,
        }
    }

    pub fn from_arg(expr: &Expr, node_arg: NodeArg) -> Self {
        let labels = extract_labels(expr);
        let (arg, arg_const) = if node_arg.is_const() {
            let mut arg = QueryValue::default();
            node_arg.resolve(&mut arg, &mut []);
            (arg, true)
        } else {
            (QueryValue::default(), false)
        };

        Self {
            keep_metric_names: false,
            labels,
            node_arg,
            arg,
            arg_const,
        }
    }

    fn set_labels_from_arg(rvs: &mut [Timeseries], arg: &Expr) {
        if let Some(labels) = extract_labels(arg) {
            for label in labels {
                rvs[0].metric_name.set_tag(&label.name, &label.value);
            }
        }
    }

    fn get_absent_timeseries(ec: &EvalConfig, arg: &Expr) -> RuntimeResult<Vec<Timeseries>> {
        // Copy tags from arg
        let mut rvs = eval_number(ec, 1.0)?;
        Self::set_labels_from_arg(&mut rvs, arg);
        Ok(rvs)
    }
}

impl ExecutableNode for AbsentTransformNode {
    fn pre_execute(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.node_arg.resolve(&mut self.arg, dependencies);
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let trace_enabled = ctx.trace_enabled();
        let span = if trace_enabled {
            trace_span!("transform", function = "absent", series = field::Empty)
        } else {
            Span::none()
        }
        .entered();

        let vec = self.arg.as_instant_vec(ec)?;
        let series_len = vec.len();

        let mut res = handle_absent(&vec, ec)?;
        if let Some(labels) = &self.labels {
            for label in labels {
                res[0].metric_name.set_tag(&label.name, &label.value);
            }
        }

        span.record("series", series_len);

        Ok(QueryValue::InstantVector(res))
    }
}
