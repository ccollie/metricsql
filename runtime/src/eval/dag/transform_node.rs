use serde::{Deserialize, Serialize};
use tracing::{field, trace_span, Span};

use metricsql::prelude::{Expr, TransformFunction};

use crate::functions::transform::{
    exec_transform_fn, extract_labels, handle_absent, TransformFuncArg,
};
use crate::{Context, EvalConfig, Labels, QueryValue, RuntimeResult};

use super::utils::resolve_args;
use super::ExecutableNode;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformNode {
    pub function: TransformFunction,
    pub keep_metric_names: bool,
    pub arg_indexes: Vec<usize>,
    #[serde(skip)]
    pub(crate) args: Vec<QueryValue>,
    /// Whether all arguments are constant and fully resolved. Note that if this is true,
    /// this node will be either be executed standalone or as the first level of a DAG evaluator.
    /// `set_dependencies` will therefore not be called.
    pub(crate) args_const: bool,
}

impl ExecutableNode for TransformNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        resolve_args(&self.arg_indexes, &mut self.args, dependencies);
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

        exec_transform_fn(self.function, &mut tfa).and_then(|res| {
            if is_tracing {
                span.record("series", res.len());
            }
            Ok(QueryValue::InstantVector(res))
        })
    }
}

impl Default for TransformNode {
    fn default() -> Self {
        Self {
            function: TransformFunction::Absent, // todo:
            args: vec![],
            keep_metric_names: false,
            arg_indexes: vec![],
            args_const: false,
        }
    }
}

// Hack. This is a copy of TransformNode. This exists solely because
// `absent` is the only transform function that requires access to the
// original expression in order to get the set of labels to apply to the result vector.
// Rather than pass around a param that's redundant in 99.9% of case we handle it here and
// hopefully keep the code cleaner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbsentTransformNode {
    pub keep_metric_names: bool,
    pub arg_index: usize,
    pub labels: Option<Labels>,
    #[serde(skip)]
    pub(crate) arg: QueryValue,
}

impl AbsentTransformNode {
    pub fn new(expr: &Expr, arg_index: usize) -> Self {
        let labels = extract_labels(expr);
        Self {
            keep_metric_names: false,
            arg_index,
            labels,
            arg: QueryValue::default(),
        }
    }
}

impl ExecutableNode for AbsentTransformNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.arg = std::mem::take(&mut dependencies[self.arg_index]);
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

impl Default for AbsentTransformNode {
    fn default() -> Self {
        Self {
            arg: QueryValue::default(),
            keep_metric_names: false,
            arg_index: 0,
            labels: None,
        }
    }
}
