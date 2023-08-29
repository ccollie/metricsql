use std::sync::Arc;

use tracing::{field, trace_span, Span};

use metricsql::ast::{DurationExpr, Expr};
use metricsql::prelude::RollupFunction;

use crate::execution::dag::utils::{
    adjust_series_by_offset, expand_single_value, resolve_at_value, resolve_rollup_handler,
};
use crate::execution::dag::{DAGNode, ExecutableNode};
use crate::execution::utils::{
    adjust_eval_range, duration_value, get_step, process_timeseries_in_parallel,
};
use crate::execution::{
    align_start_end, get_timestamps, validate_max_points_per_timeseries, Context, EvalConfig,
};
use crate::functions::rollup::{
    eval_prefuncs, get_rollup_configs, RollupHandler, MAX_SILENCE_INTERVAL,
};
use crate::{QueryValue, RuntimeResult, Timeseries};

// Node for non-selector sub-queries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubqueryNode {
    pub func: RollupFunction,
    pub(crate) func_handler: RollupHandler,
    pub expr: Expr,
    pub expr_node: Box<DAGNode>,
    pub keep_metric_names: bool,
    pub step: Option<DurationExpr>,
    pub offset: Option<DurationExpr>,
    pub window: Option<DurationExpr>,
    at: Option<i64>,
    pub at_index: Option<usize>,
    pub arg_indexes: Vec<usize>,
}

impl ExecutableNode for SubqueryNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.at = resolve_at_value(self.at_index, dependencies)?;
        self.func_handler = resolve_rollup_handler(self.func, &self.arg_indexes, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let val = if let Some(at_timestamp) = self.at {
            let mut ec_new = ec.copy_no_timestamps();
            ec_new.start = at_timestamp;
            ec_new.end = at_timestamp;

            let mut val = self.eval_without_at(ctx, &ec_new)?;
            expand_single_value(&mut val, &ec_new)?;
            val
        } else {
            self.eval_without_at(ctx, &ec)?
        };
        Ok(QueryValue::from(val))
    }
}

impl SubqueryNode {
    fn eval_without_at(
        &mut self,
        ctx: &Context,
        ec: &EvalConfig,
    ) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollup result cache here.

        let span = if ctx.trace_enabled() {
            trace_span!(
                "subquery",
                expr = self.expr.to_string().as_str(),
                rollup = self.func.name(),
                series = field::Empty,
                source_series = field::Empty,
                samples_scanned = field::Empty,
            )
        } else {
            Span::none()
        }
        .entered();

        let (offset, ec_new) = adjust_eval_range(&self.func, &self.offset, ec)?;

        let ec = ec_new;
        // todo: validate that step and window are non negative
        let step = get_step(&self.step, ec.step);
        let window = duration_value(&self.window, ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= window + MAX_SILENCE_INTERVAL + step;
        ec_sq.end += step;
        ec_sq.step = step;
        validate_max_points_per_timeseries(
            ec_sq.start,
            ec_sq.end,
            ec_sq.step,
            ec_sq.max_points_per_series,
        )?;

        // unconditionally align start and end args to step for subquery as Prometheus does.
        (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);

        let tss_sq = self.exec_expression(ctx, &ec_sq)?;
        if tss_sq.is_empty() {
            return Ok(vec![]);
        }

        let shared_timestamps = Arc::new(get_timestamps(
            ec.start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            self.func,
            &self.func_handler,
            &self.expr,
            ec.start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            min_staleness_interval,
            ec.lookback_delta,
            &shared_timestamps,
        )?;

        let (mut res, samples_scanned_total) = process_timeseries_in_parallel(
            &tss_sq,
            move |ts_sq: &Timeseries,
                  values: &mut [f64],
                  timestamps: &[i64]|
                  -> RuntimeResult<(Vec<Timeseries>, u64)> {
                let mut res = Vec::with_capacity(ts_sq.len());

                eval_prefuncs(&pre_funcs, values, timestamps);
                let mut scanned_total = 0_u64;

                for rc in rcs.iter() {
                    let (samples_scanned, mut series) = rc.process_rollup(
                        self.func,
                        &ts_sq.metric_name,
                        self.keep_metric_names,
                        values,
                        timestamps,
                        &shared_timestamps,
                    )?;

                    scanned_total += samples_scanned;
                    res.append(&mut series);
                }

                Ok((res, scanned_total))
            },
        )?;

        if !span.is_disabled() {
            span.record("series", res.len());
            span.record("source_series", tss_sq.len());
            span.record("samples_scanned", samples_scanned_total);
        }

        adjust_series_by_offset(&mut res, offset);
        Ok(res)
    }

    fn exec_expression(
        &mut self,
        ctx: &Context,
        ec: &EvalConfig,
    ) -> RuntimeResult<Vec<Timeseries>> {
        // force refresh of timestamps
        let _ = ec.get_timestamps();
        let val = self.expr_node.execute(ctx, &ec)?;
        val.get_instant_vector(ec)
    }
}
