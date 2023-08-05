use serde::{Deserialize, Serialize};
use tracing::{field, trace_span, Span};

use metricsql::prelude::AggregateModifier;
use metricsql::prelude::{AggregateFunction, AggregationExpr};

use crate::eval::dag::utils::resolve_args;
use crate::eval::dag::ExecutableNode;
use crate::functions::aggregate::{exec_aggregate_fn, AggrFuncArg};
use crate::utils::num_cpus;
use crate::{Context, EvalConfig, QueryValue, RuntimeResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateNode {
    pub function: AggregateFunction,
    pub modifier: Option<AggregateModifier>,
    pub limit: usize,
    pub(crate) arg_indexes: Vec<usize>,
    #[serde(skip)]
    pub(crate) args: Vec<QueryValue>,
    /// Whether all arguments are constant and fully resolved.
    pub(crate) args_const: bool,
}

impl ExecutableNode for AggregateNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        resolve_args(&self.arg_indexes, &mut self.args, dependencies);
        Ok(())
    }
    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let span = if ctx.trace_enabled() {
            let name = self.function.name();
            trace_span!("aggregate", name, series = field::Empty)
        } else {
            Span::none()
        }
        .entered();

        let mut afa = AggrFuncArg {
            args: if self.args_const {
                self.args.clone()
            } else {
                std::mem::take(&mut self.args)
            },
            ec,
            modifier: &self.modifier,
            limit: self.limit,
        };

        exec_aggregate_fn(self.function, &mut afa).and_then(|res| {
            if ctx.trace_enabled() {
                span.record("series", res.len());
            }
            Ok(QueryValue::InstantVector(res))
        })
    }
}

impl Default for AggregateNode {
    fn default() -> Self {
        AggregateNode {
            function: AggregateFunction::Sum,
            modifier: None,
            limit: 0,
            arg_indexes: vec![],
            args: vec![],
            args_const: false,
        }
    }
}

pub(super) fn get_timeseries_limit(aggr_expr: &AggregationExpr) -> RuntimeResult<usize> {
    // Incremental aggregates require holding only num_cpus() timeseries in memory.
    let timeseries_len = usize::from(num_cpus()?);
    let res = if aggr_expr.limit > 0 {
        // There is an explicit limit on the number of output time series.
        timeseries_len * aggr_expr.limit
    } else {
        // Increase the number of timeseries for non-empty group list: `aggr() by (something)`,
        // since each group can have own set of time series in memory.
        timeseries_len * 1000
    };

    Ok(res)
}
