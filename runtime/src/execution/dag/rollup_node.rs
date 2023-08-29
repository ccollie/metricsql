use std::sync::{Arc, Mutex};

use tracing::{field, trace_span, Span};

use lib::{AtomicCounter, RelaxedU64Counter};
use metricsql::ast::*;
use metricsql::common::Matchers;
use metricsql::functions::RollupFunction;

use crate::cache::rollup_result_cache::merge_timeseries;
use crate::execution::context::Context;
use crate::execution::dag::aggregate_node::get_timeseries_limit;
use crate::execution::dag::utils::{
    adjust_series_by_offset, expand_single_value, resolve_at_value, resolve_rollup_handler,
};
use crate::execution::dag::ExecutableNode;
use crate::execution::eval_number;
use crate::execution::utils::{adjust_eval_range, drop_stale_nans, duration_value};
use crate::execution::{get_timestamps, EvalConfig};
use crate::functions::aggregate::IncrementalAggrFuncContext;
use crate::functions::rollup::{
    eval_prefuncs, get_rollup_configs, RollupConfig, RollupHandler, MAX_SILENCE_INTERVAL,
};
use crate::functions::transform::get_absent_timeseries;
use crate::provider::{join_matchers_vec, QueryResult, QueryResults, SearchQuery};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::QueryValue;
use crate::{Timeseries, Timestamp};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RollupNode {
    /// Source expression
    pub(crate) expr: Expr,
    pub(crate) func: RollupFunction,
    pub(crate) func_handler: RollupHandler,
    pub(crate) keep_metric_names: bool,
    is_tracing: bool,
    pub(crate) metric_expr: MetricExpr,
    /// window contains optional window value from square brackets. Equivalent to `range` in
    /// prometheus terminology
    ///
    /// For example, `http_requests_total[5m]` will have Window value `5m`.
    pub window: Option<DurationExpr>,

    /// offset contains optional value from `offset` part.
    ///
    /// For example, `foobar{baz="aa"} offset 5m` will have offset value `5m`.
    pub offset: Option<DurationExpr>,

    /// step contains optional step value from square brackets. Equivalent to `resolution`
    /// in the prometheus docs
    ///
    /// For example, `foobar[1h:3m]` will have step value `3m`.
    pub step: Option<DurationExpr>,

    at: Option<i64>,
    pub(crate) at_index: Option<usize>,
    pub(crate) arg_indexes: Vec<usize>,
    pub(crate) is_incr_aggregate: bool,
}

impl ExecutableNode for RollupNode {
    fn set_dependencies(&mut self, dependencies: &mut [QueryValue]) -> RuntimeResult<()> {
        self.at = resolve_at_value(self.at_index, dependencies)?;
        self.func_handler = resolve_rollup_handler(self.func, &self.arg_indexes, dependencies)?;
        Ok(())
    }

    fn execute(&mut self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        self.is_tracing = ctx.trace_enabled();
        let _ = if self.is_tracing {
            trace_span!(
                "rollup",
                "function" = self.func.name(),
                "expr" = self.expr.to_string().as_str(),
                "series" = field::Empty
            )
        } else {
            Span::none()
        };

        if let Some(at_timestamp) = &self.at {
            let mut ec_new = ec.copy_no_timestamps();
            ec_new.start = *at_timestamp;
            ec_new.end = *at_timestamp;
            let mut tss = self.eval_without_at(ctx, &ec_new)?;

            // expand single-point tss to the original time range.
            expand_single_value(&mut tss, ec)?;

            Ok(QueryValue::InstantVector(tss))
        } else {
            let value = self.eval_without_at(ctx, ec)?;
            Ok(QueryValue::InstantVector(value))
        }
    }
}

impl RollupNode {
    pub(crate) fn new(
        function: RollupFunction,
        handler: RollupHandler,
        // expr may contain:
        // -: RollupFunc(m) if iafc is None
        // - aggrFunc(rollupFunc(m)) if iafc isn't None
        expr: &Expr,
    ) -> Self {
        Self {
            expr: expr.clone(),
            func: function,
            func_handler: handler,
            keep_metric_names: function.keep_metric_name(),
            is_incr_aggregate: false,
            is_tracing: false,
            metric_expr: Default::default(),
            window: None,
            step: None,
            offset: None,
            at: None,
            at_index: None,
            arg_indexes: vec![],
        }
    }

    fn eval_without_at(&self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let (offset, ec_new) = adjust_eval_range(&self.func, &self.offset, ec)?;

        let mut rvs = self.eval_metric_expr(ctx, &ec_new, &self.metric_expr)?;

        if self.func == RollupFunction::AbsentOverTime {
            //  rvs = aggregate_absent_over_time(ec, &self.re.expr, &rvs)?
        }

        adjust_series_by_offset(&mut rvs, offset);
        Ok(rvs)
    }
    fn eval_metric_expr(
        &self,
        ctx: &Context,
        ec: &EvalConfig,
        me: &MetricExpr,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = duration_value(&self.window, ec.step);

        let span = {
            if ctx.trace_enabled() {
                trace_span!(
                    "rollup",
                    start = ec.start,
                    end = ec.end,
                    step = ec.step,
                    window,
                    function = self.func.name(),
                    needed_memory_bytes = field::Empty
                )
            } else {
                Span::none()
            }
        }
        .entered();

        if me.is_empty() {
            return eval_number(ec, f64::NAN);
        }

        // Search for partial results in cache.

        let tss_cached: Vec<Timeseries>;
        let start: i64;
        {
            let (cached, _start) = ctx.rollup_result_cache.get(ec, &self.expr, window)?;
            tss_cached = cached.unwrap();
            start = _start;
        }

        if start > ec.end {
            // The result is fully cached.
            ctx.rollup_result_cache.full_hits.inc();
            return Ok(tss_cached);
        }

        if start > ec.start {
            ctx.rollup_result_cache.partial_hits.inc();
        } else {
            ctx.rollup_result_cache.misses.inc();
        }

        // Obtain rollup configs before fetching data from db,
        // so type errors can be caught earlier.
        let shared_timestamps = Arc::new(get_timestamps(
            start,
            ec.end,
            ec.step,
            ec.max_points_per_series,
        )?);

        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            self.func,
            &self.func_handler,
            &self.expr,
            start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            min_staleness_interval,
            ec.lookback_delta,
            &shared_timestamps,
        )?;

        let pre_func = move |values: &mut [f64], timestamps: &[i64]| {
            eval_prefuncs(&pre_funcs, values, timestamps)
        };

        // Fetch the remaining part of the result.
        let tfs = vec![Matchers::new(me.label_filters.clone())];
        let tfss = join_matchers_vec(&tfs, &ec.enforced_tag_filters);
        let mut min_timestamp = start - MAX_SILENCE_INTERVAL;
        if window > ec.step {
            min_timestamp -= &window
        } else {
            min_timestamp -= ec.step
        }
        let filters = tfss.to_vec();
        let sq = SearchQuery::new(min_timestamp, ec.end, filters, ec.max_series);
        let mut rss = ctx.process_search_query(&sq, &ec.deadline)?;

        if rss.is_empty() {
            rss.cancel();
            let dst: Vec<Timeseries> = vec![];
            let tss = merge_timeseries(tss_cached, dst, start, ec)?;
            return Ok(tss);
        }

        let mut ae: Option<&AggregationExpr> = None;
        let mut timeseries_limit = 0usize;

        if let Expr::Aggregation(_ae) = &self.expr {
            timeseries_limit = get_timeseries_limit(_ae)?;
            ae = Some(_ae);
        }

        let rollup_memory_size =
            self.reserve_rollup_memory(ctx, ec, &rss, timeseries_limit, rcs.len())?;

        defer! {
           ctx.rollup_result_cache.release_memory(rollup_memory_size).unwrap();
           span.record("needed_memory_bytes", rollup_memory_size);
        }

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = shared_timestamps.clone(); // TODO: do we need to clone ?
        let ignore_staleness = ec.no_stale_markers;
        let tss = if let Some(ae) = ae {
            self.eval_with_incremental_aggregate(
                ae,
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            )
        } else {
            self.eval_no_incremental_aggregate(
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ignore_staleness,
            )
        }?;

        merge_timeseries(tss_cached, tss, start, ec).and_then(|res| {
            ctx.rollup_result_cache.put(ec, &self.expr, window, &res)?;
            Ok(res)
        })
    }

    fn eval_with_incremental_aggregate<F>(
        &self,
        ae: &AggregationExpr,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<i64>>,
        ignore_staleness: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
    where
        F: Fn(&mut [f64], &[i64]) + Send + Sync,
    {
        let span = if self.is_tracing {
            let function = self.func.name();
            trace_span!(
                "aggregation",
                function,
                incremental = true,
                series = rss.len(),
                aggregation = ae.function.name(),
                samples_scanned = field::Empty
            )
        } else {
            Span::none()
        };

        let iafc = Arc::new(IncrementalAggrFuncContext::new(ae)?);

        struct Context<'a> {
            func: RollupFunction,
            keep_metric_names: bool,
            iafc: Arc<IncrementalAggrFuncContext<'a>>,
            rcs: Vec<RollupConfig>,
            timestamps: Arc<Vec<i64>>,
            ignore_staleness: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let mut ctx = Context {
            keep_metric_names: self.keep_metric_names,
            func: self.func,
            iafc,
            rcs,
            timestamps: Arc::clone(shared_timestamps),
            ignore_staleness,
            samples_scanned_total: Default::default(),
        };

        rss.run_parallel(
            &mut ctx,
            |ctx: Arc<&mut Context>, rs: &mut QueryResult, worker_id: u64| {
                if !ctx.ignore_staleness {
                    drop_stale_nans(ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);

                for rc in ctx.rcs.iter() {
                    let (samples_scanned, mut series) = rc.process_rollup(
                        ctx.func,
                        &rs.metric,
                        ctx.keep_metric_names,
                        &rs.values,
                        &rs.timestamps,
                        &ctx.timestamps,
                    )?;

                    for ts in series.iter_mut() {
                        ctx.iafc.update_timeseries(ts, worker_id);
                    }
                    ctx.samples_scanned_total.add(samples_scanned);
                }
                Ok(())
            },
        )?;

        let tss = ctx.iafc.finalize();

        if self.is_tracing {
            let samples_scanned = ctx.samples_scanned_total.get();
            span.record("series", tss.len());
            span.record("samples_scanned", samples_scanned);
        }

        Ok(tss)
    }

    fn eval_no_incremental_aggregate<F>(
        &self,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<Timestamp>>,
        no_stale_markers: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
    where
        F: Fn(&mut [f64], &[i64]) + Send + Sync,
    {
        let span = if self.is_tracing {
            let function = self.func.name();
            let source_series = rss.len();
            // ("aggregation", ae.name.as_str()),
            // todo: add rcs to properties
            trace_span!(
                "aggregation",
                function,
                incremental = false,
                source_series,
                series = field::Empty,
                samples_scanned = field::Empty
            )
        } else {
            Span::none()
        }
        .entered();

        struct TaskCtx<'a> {
            series: Arc<Mutex<Vec<Timeseries>>>,
            keep_metric_names: bool,
            func: RollupFunction,
            rcs: Vec<RollupConfig>,
            timestamps: &'a Arc<Vec<i64>>,
            no_stale_markers: bool,
            samples_scanned_total: RelaxedU64Counter,
        }

        let series = Arc::new(Mutex::new(Vec::with_capacity(rss.len() * rcs.len())));
        let mut ctx = TaskCtx {
            series: Arc::clone(&series),
            keep_metric_names: self.keep_metric_names,
            func: self.func,
            rcs,
            no_stale_markers,
            timestamps: shared_timestamps,
            samples_scanned_total: Default::default(),
        };

        rss.run_parallel(
            &mut ctx,
            |ctx: Arc<&mut TaskCtx>, rs: &mut QueryResult, _: u64| {
                if !ctx.no_stale_markers {
                    drop_stale_nans(ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);
                for rc in ctx.rcs.iter() {
                    let (samples_scanned, mut series) = rc.process_rollup(
                        ctx.func,
                        &rs.metric,
                        ctx.keep_metric_names,
                        &rs.values,
                        &rs.timestamps,
                        ctx.timestamps,
                    )?;

                    let mut series_vec = ctx.series.lock().unwrap();
                    series_vec.append(&mut series);
                    ctx.samples_scanned_total.add(samples_scanned);
                }
                Ok(())
            },
        )?;

        // https://users.rust-lang.org/t/how-to-move-the-content-of-mutex-wrapped-by-arc/10259/7
        let res = Arc::try_unwrap(series).unwrap().into_inner().unwrap();

        if self.is_tracing {
            let samples_scanned = ctx.samples_scanned_total.get();
            span.record("series", res.len());
            span.record("samples_scanned", samples_scanned);
        }

        Ok(res)
    }

    fn reserve_rollup_memory(
        &self,
        ctx: &Context,
        ec: &EvalConfig,
        rss: &QueryResults,
        timeseries_limit: usize,
        rcs_len: usize,
    ) -> RuntimeResult<usize> {
        // Verify timeseries fit available memory after the rollup.
        // Take into account points from tss_cached.
        let points_per_timeseries = 1 + (ec.end - ec.start) / ec.step;

        let rss_len = rss.len();
        let timeseries_len = if timeseries_limit > 0 {
            // The maximum number of output time series is limited by rss_len.
            if timeseries_limit > rss_len {
                rss_len
            } else {
                timeseries_limit
            }
        } else {
            rss_len
        };

        let rollup_points =
            mul_no_overflow(points_per_timeseries, (timeseries_len * rcs_len) as i64);
        let rollup_memory_size = mul_no_overflow(rollup_points, 16) as usize;

        let memory_limit = ctx.rollup_result_cache.memory_limit();

        if !ctx.rollup_result_cache.reserve_memory(rollup_memory_size) {
            rss.cancel();
            let msg = format!("not enough memory for processing {} data points across {} time series with {} points in each time series; \n
                                  total available memory for concurrent requests: {} bytes; requested memory: {} bytes; \n
                                  possible solutions are: reducing the number of matching time series; switching to node with more RAM; \n
                                  increasing -memory.allowedPercent; increasing `step` query arg ({})",
                              rollup_points,
                              timeseries_len * rcs_len,
                              points_per_timeseries,
                              memory_limit,
                              rollup_memory_size as u64,
                              ec.step as f64 / 1e3
            );

            return Err(RuntimeError::ResourcesExhausted(msg));
        }

        Ok(rollup_memory_size)
    }
}

/// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
///
/// Values for returned series are set to nan if at least a single tss series contains nan at that point.
/// This means that tss contains a series with non-empty results at that point.
/// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(
    ec: &EvalConfig,
    expr: &Expr,
    tss: &[Timeseries],
) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = get_absent_timeseries(ec, expr)?;
    if tss.is_empty() {
        return Ok(rvs);
    }
    for i in 0..tss[0].values.len() {
        for ts in tss {
            if ts.values[i].is_nan() {
                rvs[0].values[i] = f64::NAN;
                break;
            }
        }
    }
    Ok(rvs)
}

fn mul_no_overflow(a: i64, b: i64) -> i64 {
    a.saturating_mul(b)
}
