use rand_distr::num_traits::ops::overflowing::OverflowingMul;
use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use tracing::{field, span_enabled, trace_span, Level, Span};

use lib::{is_stale_nan, AtomicCounter, RelaxedU64Counter};
use metricsql::ast::*;
use metricsql::common::{Value, ValueType};
use metricsql::functions::{RollupFunction, Volatility};

use crate::cache::rollup_result_cache::merge_timeseries;
use crate::context::Context;
use crate::eval::arg_list::ArgList;
use crate::eval::{
    align_start_end, create_evaluator, eval_number, validate_max_points_per_timeseries,
    ExprEvaluator,
};
use crate::functions::aggregate::{Handler, IncrementalAggrFuncContext};
use crate::functions::rollup::{
    eval_prefuncs, get_rollup_configs, get_rollup_function_factory, rollup_func_keeps_metric_name,
    rollup_func_requires_config, RollupConfig, RollupHandlerEnum, RollupHandlerFactory,
    TimeseriesMap, MAX_SILENCE_INTERVAL,
};
use crate::functions::transform::get_absent_timeseries;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::{join_tag_filter_list, QueryResult, QueryResults, SearchQuery};
use crate::{get_timeseries, get_timestamps, EvalConfig, MetricName, QueryValue};
use crate::{Timeseries, Timestamp};

use super::traits::Evaluator;

pub struct RollupEvaluator {
    func: RollupFunction,
    /// Source expression
    expr: Expr,
    re: RollupExpr,
    /// Evaluator representing the expression being rolled up
    evaluator: Box<ExprEvaluator>,
    args: ArgList,
    at: Option<Box<ExprEvaluator>>,
    pub(super) incremental_aggr_handler: Option<Handler>,
    handler_factory: RollupHandlerFactory,
    keep_metric_names: bool,
    /// Max number of timeseries to return
    pub(super) timeseries_limit: usize,
    pub(super) is_incr_aggregate: bool,
}

// Init metric
// describe_counter!("rollup_result_cache_full_hits_total", "number of full request cache hits");
// describe_counter!("rollup_result_cache_partial_hits_total", "total number of partial result cache hits");

// static rollupResultCacheFullHits: _ = register_counter!("rollup_result_cache_full_hits_total");
// let rollupResultCachePartialHits = register_counter!("rollup_result_cache_partial_hits_total");
// let rollupResultCacheMiss        = register_counter!("rollup_result_cache_miss_total");

impl Value for RollupEvaluator {
    fn value_type(&self) -> ValueType {
        self.return_type()
    }
}

impl Evaluator for RollupEvaluator {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let res = self.eval_rollup(ctx, ec)?;
        Ok(QueryValue::InstantVector(res))
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }

    fn return_type(&self) -> ValueType {
        if self.at.is_some() {
            ValueType::InstantVector
        } else {
            // todo: check rollup to see
            ValueType::RangeVector
        }
    }
}

impl RollupEvaluator {
    pub fn new(re: &RollupExpr) -> RuntimeResult<Self> {
        let expr = Expr::Rollup(re.clone());
        let args: Vec<ExprEvaluator> = vec![];
        let res = Self::create_internal(RollupFunction::DefaultRollup, re, expr, args)?;
        Ok(res)
    }

    pub(crate) fn from_function(expr: &FunctionExpr) -> RuntimeResult<Self> {
        let (mut args, re) = compile_rollup_func_args(expr)?;
        // todo: tinyvec

        let func: RollupFunction;
        match RollupFunction::from_str(&expr.name) {
            Ok(rf) => func = rf,
            _ => {
                return Err(RuntimeError::UnknownFunction(format!(
                    "Expected a rollup function. Got {}",
                    expr.name
                )))
            }
        }
        let evaluator = if args.len() == 1 {
            args.remove(0)
        } else {
            match expr.arg_idx_for_optimization {
                None => {
                    // todo: this should be a bug
                    args.remove(0)
                }
                Some(idx) => {
                    // see docs for Vec::swap_remove
                    // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove
                    args.push(ExprEvaluator::default());
                    args.swap_remove(idx)
                }
            }
        };

        let mut res = Self::create_internal(func, &re, Expr::Function(expr.clone()), args)?;

        res.evaluator = Box::new(evaluator);
        Ok(res)
    }

    pub(super) fn from_metric_expression(me: MetricExpr) -> RuntimeResult<Self> {
        let re = RollupExpr::new(Expr::MetricExpression(me));
        Self::new(&re)
    }

    pub(super) fn create_internal(
        func: RollupFunction,
        re: &RollupExpr,
        expr: Expr,
        args: Vec<ExprEvaluator>,
    ) -> RuntimeResult<Self> {
        let signature = func.signature();

        let at = if let Some(re_at) = &re.at {
            let at_expr = create_evaluator(re_at)?;
            Some(Box::new(at_expr))
        } else {
            None
        };

        let keep_metric_names = get_keep_metric_names(&expr);
        let nrf = get_rollup_function_factory(func);
        let evaluator = Box::new(create_evaluator(&expr)?);
        let args = ArgList::from(&signature, args);

        let res = RollupEvaluator {
            func,
            expr,
            re: re.clone(),
            evaluator,
            args,
            at,
            incremental_aggr_handler: None,
            handler_factory: nrf,
            keep_metric_names,
            timeseries_limit: 0,
            is_incr_aggregate: false,
        };

        Ok(res)
    }

    /// Some rollup functions require config, hence in rollup_fns we
    /// first obtain a factory which configures the final function used
    /// to process the rollup. For example `hoeffding_bound_upper(phi, series_selector[d])`,
    /// where `phi` is used to adjust the runtime behaviour of the function.
    ///
    /// However the majority of functions simply take a vector and process the
    /// rollup, so for those cases we perform a small optimization by calling the factory
    /// and storing the result.
    ///
    /// It is also possible to optimize in cases where the non-selector inputs are all scalar
    /// or string (meaning constant), meaning the result of the factory call is essentially
    /// idempotent
    fn get_rollup_handler(
        func: &RollupFunction,
        arg_list: &mut ArgList,
        _factory: RollupHandlerFactory,
    ) -> Option<RollupHandlerEnum> {
        if rollup_func_requires_config(func) {
            if !arg_list.all_const() {
                return None;
            }
        }
        None
        // todo: unfinished
    }

    // expr may contain:
    // -: RollupFunc(m) if iafc is None
    // - aggrFunc(rollupFunc(m)) if iafc isn't None
    fn eval_rollup(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo(perf): if function is not volatile and the non metric args are const, we can store
        // the results of the nrf call since the result won't change, hence sparing the call to eval
        // the arg list. For ex `quantile(0.95, latency{func="make-widget"})`. We know which is
        let params = self.args.eval(ctx, ec)?;
        let rf = (self.handler_factory)(&params)?;

        if self.at.is_none() {
            return self.eval_without_at(ctx, ec, &rf);
        }

        let at_timestamp = self.get_at_timestamp(ctx, ec, &self.at.as_ref().unwrap())?;
        let mut ec_new = ec.copy_no_timestamps();
        ec_new.start = at_timestamp;
        ec_new.end = at_timestamp;
        let mut tss = self.eval_without_at(ctx, &mut ec_new, &rf)?;

        // expand single-point tss to the original time range.
        let timestamps = ec.timestamps();
        for ts in tss.iter_mut() {
            ts.timestamps = Arc::clone(&timestamps);
            ts.values = vec![ts.values[0]; timestamps.len()];
        }

        return Ok(tss);
    }

    fn get_at_timestamp(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
        evaluator: &ExprEvaluator,
    ) -> RuntimeResult<i64> {
        match evaluator.eval(ctx, ec) {
            Err(err) => {
                let msg = format!("cannot evaluate `@` modifier: {:?}", err);
                return Err(RuntimeError::from(msg));
            }
            Ok(tss_at) => {
                match tss_at {
                    QueryValue::InstantVector(v) => {
                        if v.len() != 1 {
                            let msg = format!("`@` modifier must return a single series; it returns {} series instead", v.len());
                            return Err(RuntimeError::from(msg));
                        }
                        let ts = &v[0];
                        if ts.values.len() > 1 {
                            let msg = format!("`@` modifier must return a single value; it returns {} series instead", ts.values.len());
                            return Err(RuntimeError::from(msg));
                        }
                        Ok((ts.values[0] * 1000_f64) as i64)
                    }
                    _ => {
                        let val = tss_at.get_int()?;
                        Ok(val * 1000_i64)
                    }
                }
            }
        }
    }

    #[inline]
    fn get_offset(&self, step: i64) -> i64 {
        if let Some(ofs) = &self.re.offset {
            ofs.value(step)
        } else {
            0
        }
    }

    #[inline]
    fn get_window(&self, step: i64) -> i64 {
        if let Some(win) = &self.re.window {
            win.value(step)
        } else {
            0
        }
    }

    #[inline]
    fn get_step(&self, step: i64) -> i64 {
        if let Some(v) = &self.re.step {
            let res = v.value(step);
            if res == 0 {
                step
            } else {
                res
            }
        } else {
            0
        }
    }

    #[inline]
    fn adjust_eval_range<'a>(
        &self,
        ec: &'a EvalConfig,
    ) -> RuntimeResult<(i64, Cow<'a, EvalConfig>)> {
        let offset: i64 = self.get_offset(ec.step);

        let mut adjustment = 0 - offset;
        if self.func == RollupFunction::RollupCandlestick {
            // Automatically apply `offset -step` to `rollup_candlestick` function
            // in order to obtain expected OHLC results.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
            adjustment += ec.step
        }

        if adjustment != 0 {
            let mut result = ec.copy_no_timestamps();
            result.start += adjustment;
            result.end += adjustment;
            result.ensure_timestamps()?;
            // There is no need in calling adjust_start_end() on ec_new if ec_new.may_cache is set to true,
            // since the time range alignment has been already performed by the caller,
            // so cache hit rate should be quite good.
            // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
            Ok((offset, Cow::Owned(result)))
        } else {
            Ok((offset, Cow::Borrowed(ec)))
        }
    }

    fn eval_without_at(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
        rollup_func: &RollupHandlerEnum,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let (offset, ec_new) = self.adjust_eval_range(ec)?;

        let mut rvs = match &*self.re.expr {
            Expr::MetricExpression(me) => {
                self.eval_with_metric_expr(ctx, &ec_new, me, rollup_func)?
            }
            _ => {
                // todo: do this check on Evaluator construction
                if self.is_incr_aggregate {
                    let msg = format!(
                        "BUG:iafc must be None for rollup {} over subquery {}",
                        self.func, &self.re
                    );
                    return Err(RuntimeError::from(msg));
                }
                self.eval_with_subquery(ctx, &ec_new, rollup_func)?
            }
        };

        if self.func == RollupFunction::AbsentOverTime {
            rvs = aggregate_absent_over_time(ec, &self.re.expr, &rvs)
        }

        if offset != 0 && rvs.len() > 0 {
            // Make a copy of timestamps, since they may be used in other values.
            let src_timestamps = &rvs[0].timestamps;
            let dst_timestamps = src_timestamps.iter().map(|x| x + offset).collect();
            let shared = Arc::new(dst_timestamps);
            for ts in rvs.iter_mut() {
                ts.timestamps = Arc::clone(&shared);
            }
        }

        Ok(rvs)
    }

    fn eval_with_subquery(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
        rollup_func: &RollupHandlerEnum,
    ) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollup result cache here.

        let span = if ctx.trace_enabled() {
            let function = self.func.name();
            trace_span!(
                "subquery",
                function,
                series = field::Empty,
                source_series = field::Empty,
                samples_scanned = field::Empty,
            )
        } else {
            Span::none()
        }
        .entered();

        let step = self.get_step(ec.step);
        let window = self.get_window(ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= window + MAX_SILENCE_INTERVAL + step;
        ec_sq.end += step;
        ec_sq.step = step;
        validate_max_points_per_timeseries(
            ec_sq.start,
            ec_sq.end,
            ec_sq.step,
            ec.max_points_per_series,
        )?;

        // unconditionally align start and end args to step for subquery as Prometheus does.
        (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);
        let tss_sq = self.evaluator.eval(ctx, &ec_sq)?;

        let tss_sq = tss_sq.as_instant_vec(&ec)?;
        if tss_sq.len() == 0 {
            return Ok(vec![]);
        }

        let shared_timestamps = ec.timestamps();
        let min_staleness_interval = ctx.config.min_staleness_interval.num_milliseconds() as usize;
        let (rcs, pre_funcs) = get_rollup_configs(
            &self.func,
            rollup_func,
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

        let keep_metric_names = self.keep_metric_names;
        let func = self.func;

        let (res, samples_scanned_total) = do_parallel(
            &tss_sq,
            move |ts_sq: &Timeseries,
                  values: &mut [f64],
                  timestamps: &[i64]|
                  -> RuntimeResult<(Vec<Timeseries>, u64)> {
                let mut res: Vec<Timeseries> = Vec::with_capacity(ts_sq.len());

                eval_prefuncs(&pre_funcs, values, timestamps);
                let mut scanned_total = 0_u64;

                for rc in rcs.iter() {
                    match new_timeseries_map(
                        &func,
                        keep_metric_names,
                        &shared_timestamps,
                        &ts_sq.metric_name,
                    ) {
                        Some(tsm) => {
                            rc.do_timeseries_map(&tsm, values, timestamps)?;
                            tsm.as_ref().borrow_mut().append_timeseries_to(&mut res);
                            continue;
                        }
                        _ => {}
                    }

                    let mut ts: Timeseries = Default::default();

                    let scanned_samples = do_rollup_for_timeseries(
                        keep_metric_names,
                        rc,
                        &mut ts,
                        &ts_sq.metric_name,
                        &values,
                        &timestamps,
                        &shared_timestamps,
                    )?;

                    scanned_total += scanned_samples;

                    res.push(ts);
                }

                Ok((res, scanned_total))
            },
        )?;

        if !span.is_disabled() {
            span.record("series", res.len());
            span.record("source_series", tss_sq.len());
            span.record("samples_scanned", samples_scanned_total);
        }

        Ok(res)
    }

    fn eval_with_metric_expr(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
        me: &MetricExpr,
        rollup_func: &RollupHandlerEnum,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = self.get_window(ec.step);

        let is_tracing = ctx.trace_enabled();
        let span = {
            if is_tracing {
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
            return Ok(eval_number(ec, f64::NAN));
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
            &self.func,
            rollup_func,
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
        let tfs = vec![me.label_filters.clone()];
        let tfss = join_tag_filter_list(&tfs, &ec.enforced_tag_filters);
        let mut min_timestamp = start - MAX_SILENCE_INTERVAL;
        if window > ec.step {
            min_timestamp -= &window
        } else {
            min_timestamp -= ec.step
        }
        let filters = tfss.to_vec();
        let sq = SearchQuery::new(min_timestamp, ec.end, filters, ec.max_series);
        let mut rss = ctx.process_search_query(&sq, &ec.deadline)?;
        let rss_len = rss.len();
        if rss_len == 0 {
            rss.cancel();
            let dst: Vec<Timeseries> = vec![];
            let tss = merge_timeseries(tss_cached, dst, start, ec)?;
            return Ok(tss);
        }

        let rollup_memory_size = self.reserve_rollup_memory(ctx, ec, &mut rss, rcs.len())?;

        defer! {
           ctx.rollup_result_cache.release_memory(rollup_memory_size).unwrap();
           span.record("needed_memory_bytes", rollup_memory_size);
        }

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = Arc::new(shared_timestamps);
        let tss = match &self.expr {
            Expr::Aggregation(ae) => self.eval_with_incremental_aggregate(
                &ae,
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
            ),
            _ => self.eval_no_incremental_aggregate(
                &mut rss,
                rcs,
                pre_func,
                &shared_timestamps,
                ec.no_stale_markers,
            ),
        };

        match tss {
            Ok(v) => match merge_timeseries(tss_cached, v, start, ec) {
                Ok(res) => {
                    ctx.rollup_result_cache.put(ec, &self.expr, window, &res)?;
                    Ok(res)
                }
                Err(err) => Err(err),
            },
            Err(e) => Err(e),
        }
    }

    fn reserve_rollup_memory(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
        rss: &QueryResults,
        rcs_len: usize,
    ) -> RuntimeResult<usize> {
        // Verify timeseries fit available memory after the rollup.
        // Take into account points from tss_cached.
        let points_per_timeseries = 1 + (ec.end - ec.start) / ec.step;

        let rss_len = rss.len();
        let timeseries_len = if self.timeseries_limit > 0 {
            // The maximum number of output time series is limited by rss_len.
            if self.timeseries_limit > rss_len {
                rss_len
            } else {
                self.timeseries_limit
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

    fn eval_with_incremental_aggregate<F>(
        &self,
        ae: &AggregationExpr,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<i64>>,
    ) -> RuntimeResult<Vec<Timeseries>>
    where
        F: Fn(&mut [f64], &[i64]) -> () + Send + Sync,
    {
        let is_tracing = span_enabled!(Level::TRACE);
        let span = if is_tracing {
            let function = self.func.name();
            let span = trace_span!(
                "rollup",
                function,
                incremental = true,
                series = rss.len(),
                aggregation = ae.name,
                samples_scanned = field::Empty
            );
            span
        } else {
            Span::none()
        };

        struct Context<'a> {
            func: &'a RollupFunction,
            keep_metric_names: bool,
            iafc: Mutex<IncrementalAggrFuncContext<'a>>,
            rcs: Vec<RollupConfig>,
            timestamps: &'a Arc<Vec<i64>>,
            samples_scanned_total: RelaxedU64Counter,
        }

        let callbacks = self.incremental_aggr_handler.as_ref().unwrap();
        let iafc = Mutex::new(IncrementalAggrFuncContext::new(ae, callbacks));

        let mut ctx = Context {
            keep_metric_names: self.keep_metric_names,
            func: &self.func,
            iafc,
            rcs,
            timestamps: &shared_timestamps,
            samples_scanned_total: Default::default(),
        };

        rss.run_parallel(
            &mut ctx,
            |ctx: Arc<&mut Context>, rs: &mut QueryResult, worker_id: u64| {
                drop_stale_nans(&ctx.func, &mut rs.values, &mut rs.timestamps);
                pre_func(&mut rs.values, &rs.timestamps);

                for rc in ctx.rcs.iter() {
                    match new_timeseries_map(
                        &ctx.func,
                        ctx.keep_metric_names,
                        &ctx.timestamps,
                        &rs.metric_name,
                    ) {
                        Some(tsm) => {
                            rc.do_timeseries_map(&tsm, &rs.values, &rs.timestamps)?;
                            let iafc = ctx.iafc.lock().unwrap();
                            for ts in tsm.as_ref().borrow_mut().values_mut() {
                                iafc.update_timeseries(ts, worker_id)?;
                            }
                            continue;
                        }
                        _ => {}
                    }

                    let mut ts = get_timeseries();
                    let samples_scanned = do_rollup_for_timeseries(
                        ctx.keep_metric_names,
                        rc,
                        &mut ts,
                        &rs.metric_name,
                        &rs.values,
                        &rs.timestamps,
                        &ctx.timestamps,
                    )?;

                    ctx.samples_scanned_total.add(samples_scanned);
                    // todo: return result rather than unwrap
                    let iafc = ctx.iafc.lock().unwrap();
                    iafc.update_timeseries(&mut ts, worker_id).unwrap();
                }
                Ok(())
            },
        )?;

        let mut iafc = ctx.iafc.lock().unwrap();
        let tss = iafc.finalize();

        if is_tracing {
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
        F: Fn(&mut [f64], &[i64]) -> () + Send + Sync,
    {
        let is_tracing = span_enabled!(Level::TRACE); //
        let span = if is_tracing {
            let function = self.func.name();
            let source_series = rss.len();
            // ("aggregation", ae.name.as_str()),
            // todo: add rcs to properties
            trace_span!(
                "rollup",
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
                    drop_stale_nans(&ctx.func, &mut rs.values, &mut rs.timestamps);
                }
                pre_func(&mut rs.values, &rs.timestamps);
                for rc in ctx.rcs.iter() {
                    match new_timeseries_map(
                        &ctx.func,
                        ctx.keep_metric_names,
                        &ctx.timestamps,
                        &rs.metric_name,
                    ) {
                        Some(tsm) => {
                            rc.do_timeseries_map(&tsm, &rs.values, &rs.timestamps)?;
                            let mut tss = ctx.series.lock().unwrap();
                            tsm.as_ref().borrow_mut().append_timeseries_to(&mut tss);
                        }
                        _ => {
                            let mut ts: Timeseries = Timeseries::default();
                            let samples_scanned = do_rollup_for_timeseries(
                                ctx.keep_metric_names,
                                rc,
                                &mut ts,
                                &rs.metric_name,
                                &rs.values,
                                &rs.timestamps,
                                ctx.timestamps,
                            )?;

                            ctx.samples_scanned_total.add(samples_scanned);

                            let mut tss = ctx.series.lock().unwrap();
                            tss.push(ts);
                        }
                    }
                }
                Ok(())
            },
        )?;

        // https://users.rust-lang.org/t/how-to-move-the-content-of-mutex-wrapped-by-arc/10259/7
        let res = Arc::try_unwrap(series).unwrap().into_inner().unwrap();

        if is_tracing {
            let samples_scanned = ctx.samples_scanned_total.get();
            span.record("series", res.len());
            span.record("samples_scanned", samples_scanned);
        }

        Ok(res)
    }
}

#[inline]
fn new_timeseries_map(
    func: &RollupFunction,
    keep_metric_names: bool,
    shared_timestamps: &Arc<Vec<Timestamp>>,
    mn: &MetricName,
) -> Option<Rc<RefCell<TimeseriesMap>>> {
    match TimeseriesMap::new(&func, keep_metric_names, shared_timestamps, &mn) {
        None => None,
        Some(map) => Some(Rc::new(RefCell::new(map))),
    }
}

pub(super) fn compile_rollup_func_args(
    fe: &FunctionExpr,
) -> RuntimeResult<(Vec<ExprEvaluator>, RollupExpr)> {
    let rollup_arg_idx = fe.arg_idx_for_optimization;
    if rollup_arg_idx.is_none() {
        let err = format!(
            "Bug: can't find source arg for rollup function {}. Expr: {}",
            fe.name, fe
        );
        return Err(RuntimeError::ArgumentError(err));
    }

    let mut re: RollupExpr = RollupExpr::new(Expr::from(""));

    let arg_idx = rollup_arg_idx.unwrap();

    let mut args: Vec<ExprEvaluator> = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == arg_idx {
            re = get_rollup_expr_arg(arg)?;
            args.push(create_evaluator(&Expr::Rollup(re.clone()))?);
            continue;
        }
        args.push(create_evaluator(&*arg)?);
    }

    return Ok((args, re));
}

// todo: COW. This has a lot of clones
fn get_rollup_expr_arg(arg: &Expr) -> RuntimeResult<RollupExpr> {
    let mut re: RollupExpr = match arg {
        Expr::Rollup(re) => re.clone(),
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            RollupExpr::new(arg.clone())
        }
    };

    if !re.for_subquery() {
        // Return standard rollup if it doesn't contain subquery.
        return Ok(re);
    }

    return match re.expr.deref() {
        Expr::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]

            // TODO: avoid clone below. Can we just do an into() ?
            let arg = Expr::Rollup(RollupExpr::new(Expr::MetricExpression(me.clone())));

            match FunctionExpr::default_rollup(arg) {
                Err(e) => return Err(RuntimeError::General(format!("{:?}", e))),
                Ok(fe) => {
                    re.expr = Box::new(Expr::Function(fe));
                    Ok(re)
                }
            }
        }
        _ => {
            // arg contains subquery.
            Ok(re)
        }
    };
}

pub(super) fn adjust_eval_range(
    ec: &EvalConfig,
    offset: i64,
    func: RollupFunction,
) -> RuntimeResult<Cow<EvalConfig>> {
    let mut adjustment = 0 - offset;
    if func == RollupFunction::RollupCandlestick {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        adjustment += ec.step
    }

    if adjustment != 0 {
        let mut result = ec.copy_no_timestamps();
        result.start += adjustment;
        result.end += adjustment;
        result.ensure_timestamps()?;
        // There is no need in calling adjust_start_end() on ec_new if ec_new.may_cache is set to true,
        // since the time range alignment has been already performed by the caller,
        // so cache hit rate should be quite good.
        // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
        Ok(Cow::Owned(result))
    } else {
        Ok(Cow::Borrowed(ec))
    }
}

/// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
///
/// Values for returned series are set to nan if at least a single tss series contains nan at that point.
/// This means that tss contains a series with non-empty results at that point.
/// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(ec: &EvalConfig, expr: &Expr, tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs = get_absent_timeseries(ec, expr);
    if tss.len() == 0 {
        return rvs;
    }
    for i in 0..tss[0].values.len() {
        for ts in tss {
            if ts.values[i].is_nan() {
                rvs[0].values[i] = f64::NAN;
                break;
            }
        }
    }
    return rvs;
}

fn get_keep_metric_names(expr: &Expr) -> bool {
    // todo: move to optimize stage. put result in ast node
    return match expr {
        Expr::BinaryOperator(be) => return be.keep_metric_names,
        Expr::Aggregation(ae) => {
            if ae.keep_metric_names {
                return rollup_func_keeps_metric_name(&ae.name);
            }
            false
        }
        Expr::Function(fe) => {
            if fe.keep_metric_names {
                return rollup_func_keeps_metric_name(&fe.name);
            }
            false
        }
        _ => false,
    };
}

fn do_parallel<F>(tss: &Vec<Timeseries>, f: F) -> RuntimeResult<(Vec<Timeseries>, u64)>
where
    F: Fn(&Timeseries, &mut [f64], &[i64]) -> RuntimeResult<(Vec<Timeseries>, u64)> + Send + Sync,
{
    let res = tss
        .par_iter()
        .map(|ts| {
            let len = ts.values.len();
            // todo(perf): use object pool for these
            let mut values: Vec<f64> = Vec::with_capacity(len);
            let mut timestamps: Vec<i64> = Vec::with_capacity(len);

            // todo(perf): have param for if values have NaNs
            remove_nan_values(&mut values, &mut timestamps, &ts.values, &ts.timestamps);

            f(ts, &mut values, &mut timestamps)
        })
        .collect::<Vec<_>>();

    let mut series: Vec<Timeseries> = Vec::with_capacity(tss.len());
    let mut sample_total = 0_u64;
    for r in res.into_iter() {
        match r {
            Err(e) => return Err(e),
            Ok((timeseries, sample_count)) => {
                sample_total += sample_count;
                series.extend::<Vec<Timeseries>>(timeseries.into())
            }
        }
    }

    return Ok((series, sample_total));
}

fn remove_nan_values(
    dst_values: &mut Vec<f64>,
    dst_timestamps: &mut Vec<i64>,
    values: &[f64],
    timestamps: &[i64],
) {
    let mut has_nan = false;
    for v in values {
        if v.is_nan() {
            has_nan = true
        }
    }

    if !has_nan {
        // Fast path - no NaNs.
        dst_values.extend_from_slice(&values);
        dst_timestamps.extend_from_slice(&timestamps);
        return;
    }

    // Slow path - remove NaNs.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        dst_values.push(*v);
        dst_timestamps.push(timestamps[i])
    }
}

fn do_rollup_for_timeseries(
    keep_metric_names: bool,
    rc: &RollupConfig,
    ts_dst: &mut Timeseries,
    mn_src: &MetricName,
    values_src: &[f64],
    timestamps_src: &[i64],
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<u64> {
    ts_dst.metric_name.copy_from(mn_src);
    if rc.tag_value.len() > 0 {
        ts_dst.metric_name.set_tag("rollup", &rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_metric_group();
    }
    let samples_scanned = rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = Arc::clone(&shared_timestamps);

    Ok(samples_scanned)
}

fn mul_no_overflow(a: i64, b: i64) -> i64 {
    let (res, overflow) = a.overflowing_mul(b);
    return if overflow { i64::MAX } else { res };
}

pub(crate) fn drop_stale_nans(
    func: &RollupFunction,
    values: &mut Vec<f64>,
    timestamps: &mut Vec<i64>,
) {
    if *func == RollupFunction::DefaultRollup || *func == RollupFunction::StaleSamplesOverTime {
        // do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return;
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = values.iter().any(|x| is_stale_nan(*x));

    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return;
    }

    // Slow path: drop Prometheus staleness marks from values.
    let mut k = 0;
    for i in 0..values.len() {
        let v = values[i];
        if !is_stale_nan(v) {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
    }

    values.truncate(k);
    timestamps.truncate(k);
}
