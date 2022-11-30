use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use rayon::prelude::IntoParallelRefIterator;

use lib::is_stale_nan;
use metricsql::ast::*;
use metricsql::functions::{BuiltinFunction, DataType, RollupFunction, Volatility};

use crate::{EvalConfig, get_timeseries, get_timestamps, MetricName};
use crate::cache::rollup_result_cache::merge_timeseries;
use crate::context::Context;
use crate::eval::{
    align_start_end,
    create_evaluator,
    eval_number,
    ExprEvaluator,
    validate_max_points_per_timeseries,
};
use crate::eval::arg_list::ArgList;
use crate::functions::aggregate::{IncrementalAggrFuncCallbacks, IncrementalAggrFuncContext};
use crate::functions::rollup::{
    eval_prefuncs, get_rollup_configs, get_rollup_function_factory,
    MAX_SILENCE_INTERVAL, rollup_func_keeps_metric_name, rollup_func_requires_config,
    RollupConfig, RollupHandlerEnum, RollupHandlerFactory, TimeseriesMap
};
use crate::functions::transform::get_absent_timeseries;
use crate::functions::types::AnyValue;
use crate::rayon::iter::ParallelIterator;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::{join_tag_filterss, QueryResult, QueryResults, SearchQuery};
use crate::timeseries::Timeseries;
use crate::traits::Timestamp;

use super::traits::Evaluator;

pub(crate) struct IncrementalAggrFuncOptions {
    pub callbacks: &'static IncrementalAggrFuncCallbacks,
}

pub struct RollupEvaluator {
    func: RollupFunction,
    /// Source expression
    expr: Expression,
    re: RollupExpr,
    /// Evaluator representing the expression being rolled up
    evaluator: Box<ExprEvaluator>,
    args: ArgList,
    at: Option<Box<ExprEvaluator>>,
    pub(super) incremental_aggr_opts: Option<IncrementalAggrFuncOptions>,
    nrf: RollupHandlerFactory,
    keep_metric_names: bool,
    /// Max number of timeseries to return
    pub(super) timeseries_limit: usize,
    pub(super) is_incr_aggregate: bool
}

// Init metric
// describe_counter!("vm_rollup_result_cache_full_hits_total", "number of full request cache hits");
// describe_counter!("vm_rollup_result_cache_partial_hits_total", "total number of partial result cache hits");

// static rollupResultCacheFullHits: _ = register_counter!("vm_rollup_result_cache_full_hits_total");
// let rollupResultCachePartialHits = register_counter!("vm_rollup_result_cache_partial_hits_total");
// let rollupResultCacheMiss        = register_counter!("vm_rollup_result_cache_miss_total");

impl Evaluator for RollupEvaluator {
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        let res = self.eval_rollup(ctx, ec)?;
        Ok(AnyValue::InstantVector(res))
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }
    
    fn return_type(&self) -> DataType {
        if self.at.is_some() {
            DataType::InstantVector
        } else {
            // todo: check rollup to see
            DataType::RangeVector
        }
    }
}

impl RollupEvaluator {
    pub fn new(re: &RollupExpr) -> RuntimeResult<Self> {
        let expr = Expression::Rollup(re.clone());
        let args: Vec<ExprEvaluator> = vec![];
        let res = Self::create_internal(RollupFunction::DefaultRollup, re, expr, args)?;
        Ok(res)
    }

    pub fn from_function(expr: &FuncExpr) -> RuntimeResult<Self> {
        let (mut args, re) = compile_rollup_func_args(expr)?;
        // todo: tinyvec

        let func: RollupFunction;
        match RollupFunction::from_str(&expr.name) {
            Ok(rf) => func = rf,
            _ => {
                return Err( RuntimeError::UnknownFunction(
                    format!("Expected a rollup function. Got {}", expr.name)
                ))
            }
        }
        let evaluator = if args.len() == 1 {
            args.remove(0)
        } else {
            let rollup_index = expr.get_arg_idx_for_optimization();
            match rollup_index {
                None => {
                    // todo: this should be a bug
                    args.remove(0)
                },
                Some(idx) => {
                    // see docs fpr Vec::swap_remove
                    // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove
                    args.push(ExprEvaluator::default());
                    args.swap_remove(idx)
                }
            }
        };

        let mut res = Self::create_internal(
            func,
            &re,
            Expression::Function(expr.clone()),
            args
        )?;

        res.evaluator = Box::new(evaluator);
        Ok(res)
    }

    pub(super) fn from_metric_expression(me: MetricExpr) -> RuntimeResult<Self> {
        let re = RollupExpr::new(Expression::MetricExpression(me));
        Self::new(&re)
    }

    pub(super) fn create_internal(func: RollupFunction,
                                  re: &RollupExpr,
                                  expr: Expression,
                                  args: Vec<ExprEvaluator>) -> RuntimeResult<Self> {

        let signature = func.signature();

        let at = if let Some(re_at) = &re.at {
            let at_expr = create_evaluator(re_at)?;
            Some( Box::new(at_expr) )
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
            incremental_aggr_opts: None,
            nrf,
            keep_metric_names,
            timeseries_limit: 0,
            is_incr_aggregate: false
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
    ///  It is also possible to optimize in cases where the non-selector inputs are all scalar
    ///  or string (meaning constant), meaning the result of the factory call is essentially
    ///  idempotent
    fn get_rollup_handler(func: &RollupFunction, arg_list: &mut ArgList, _factory: RollupHandlerFactory) -> Option<RollupHandlerEnum> {
        if rollup_func_requires_config(func) {
            if !arg_list.all_const() {
                return None
            }
        }
        None
    }

    // expr may contain:
    // -: RollupFunc(m) if iafc is None
    // - aggrFunc(rollupFunc(m)) if iafc isn't None
    fn eval_rollup(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo(perf): if function is not volatile and the non metric args are const, we can store
        // the results of the nrf call since the result won't change, hence sparing the call to eval
        // the arg list. For ex `quantile(0.95, latency{func="make-widget"})`. We know which is
        let params = self.args.eval(ctx, ec)?;
        let rf = (self.nrf)(&params)?;

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

    #[inline]
    fn get_at_timestamp(&self, ctx: &Arc<&Context>, ec: &EvalConfig, evaluator: &ExprEvaluator) -> RuntimeResult<i64> {
        match evaluator.eval(ctx, ec) {
            Err(err) => {
                let msg = format!("cannot evaluate `@` modifier: {:?}", err);
                return Err(RuntimeError::from(msg));
            }
            Ok(tss_at) => {
                match tss_at {
                    AnyValue::InstantVector(v) => {
                        if v.len() != 1 {
                            let msg = format!("`@` modifier must return a single series; it returns {} series instead", v.len());
                            return Err(RuntimeError::from(msg));
                        }
                        let ts = v.get(0).unwrap();
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
            ofs.duration(step)
        } else {
            0
        }
    }

    #[inline]
    fn get_window(&self, step: i64) -> i64 {
        if let Some(win) = &self.re.window {
            win.duration(step)
        } else {
            0
        }
    }

    #[inline]
    fn get_step(&self, step: i64) -> i64 {
        if let Some(v) = &self.re.step {
            let res = v.duration(step);
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
    fn adjust_eval_range<'a>(&self, ec: &'a EvalConfig) -> RuntimeResult<(i64, Cow<'a, EvalConfig>)> {
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

    fn eval_without_at(&self, ctx: &Arc<&Context>, ec: &EvalConfig, rollup_func: &RollupHandlerEnum) -> RuntimeResult<Vec<Timeseries>> {

        let (offset, ec_new) = self.adjust_eval_range(ec)?;

        let mut rvs = match &*self.re.expr {
            Expression::MetricExpression(me) => {
                self.eval_with_metric_expr(ctx, &ec_new, me, rollup_func)?
            }
            _ => {
                if self.is_incr_aggregate {
                    let msg = format!("BUG: iafc must be None for rollup {} over subquery {}", self.func, &self.re);
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

    fn eval_with_subquery(&self,
                          ctx: &Arc<&Context>,
                          ec: &EvalConfig,
                          rollup_func: &RollupHandlerEnum) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollup result cache here.

        let step = self.get_step(ec.step);
        let window = self.get_window(ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= window + MAX_SILENCE_INTERVAL + step;
        ec_sq.end += step;
        ec_sq.step = step;
        validate_max_points_per_timeseries(ec_sq.start, ec_sq.end, ec_sq.step, ec.max_points_per_series)?;

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
            &shared_timestamps)?;

        let keep_metric_names = self.keep_metric_names;
        let func = self.func;

        do_parallel(&tss_sq,  move |ts_sq: &Timeseries, values: &mut [f64], timestamps: &[i64]| -> RuntimeResult<Vec<Timeseries>> {

            let mut res: Vec<Timeseries> = Vec::with_capacity(ts_sq.len());

            eval_prefuncs(&pre_funcs, values, timestamps);
            for rc in rcs.iter() {
                match new_timeseries_map(&func, keep_metric_names, &shared_timestamps, &ts_sq.metric_name) {
                    Some(tsm) => {
                        rc.do_timeseries_map(&tsm, values, timestamps)?;
                        tsm.as_ref().borrow_mut().append_timeseries_to(&mut res);
                        continue;
                    }
                    _ => {}
                }

                let mut ts: Timeseries = Default::default();

                do_rollup_for_timeseries(
                    keep_metric_names,
                    rc,
                    &mut ts,
                    &ts_sq.metric_name,
                    &values,
                    &timestamps,
                    &shared_timestamps)?;

                res.push(ts);
            }

            Ok(res)
        })
    }


    fn eval_with_metric_expr(
        &self,
        ctx: &Arc<&Context>,
        ec: &EvalConfig,
        me: &MetricExpr,
        rollup_func: &RollupHandlerEnum,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = self.get_window(ec.step);

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
            // rollupResultCacheFullHits.inc();
            return Ok(tss_cached);
        }

        // if start > ec.start {
        //     rollupResultCachePartialHits.inc();
        // } else {
        //     rollupResultCacheMiss.Inc();
        // }

        // Obtain rollup configs before fetching data from db,
        // so type errors can be caught earlier.
        let shared_timestamps = Arc::new(
            get_timestamps(start, ec.end, ec.step, ec.max_points_per_series)?
        );

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
            &shared_timestamps)?;

        let pre_func = move |values: &mut [f64], timestamps: &[i64]| {
            eval_prefuncs(&pre_funcs, values, timestamps)
        };

        // Fetch the remaining part of the result.
        let tfs = vec![me.label_filters.clone()];
        let tfss = join_tag_filterss(&tfs, &ec.enforced_tag_filterss);
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

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = Arc::new(shared_timestamps);
        let tss = match &self.expr {
            Expression::Aggregation(ae) => {
                self.eval_with_incremental_aggregate(
                    &ae,
                    &mut rss,
                    rcs,
                    pre_func,
                    &shared_timestamps)
            },
            _ => {
                self.eval_no_incremental_aggregate(
                    &mut rss,
                    rcs,
                    pre_func,
                    &shared_timestamps,
                    ec.no_stale_markers)
            }
        };

        match tss {
            Ok(v) => {
                match merge_timeseries(tss_cached, v, start, ec) {
                    Ok(res) => {
                        ctx.rollup_result_cache.release_memory(rollup_memory_size)?;
                        ctx.rollup_result_cache.put(ec, &self.expr, window, &res)?;
                        Ok(res)
                    },
                    Err(err) => {
                        ctx.rollup_result_cache.release_memory(rollup_memory_size)?;
                        Err(err)
                    }
                }
            }
            Err(e) => {
                ctx.rollup_result_cache.release_memory(rollup_memory_size)?;
                Err(e)
            }
        }
    }

    fn reserve_rollup_memory(&self, ctx: &Arc<&Context>, ec: &EvalConfig, rss: &QueryResults, rcs_len: usize) -> RuntimeResult<usize> {
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

        let rollup_points = mul_no_overflow(points_per_timeseries, (timeseries_len * rcs_len) as i64);
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

            return Err(RuntimeError::from(msg));
        }

        Ok(rollup_memory_size)
    }

    fn eval_with_incremental_aggregate<F>(&self,
                                          ae: &AggrFuncExpr,
                                          rss: &mut QueryResults,
                                          rcs: Vec<RollupConfig>,
                                          pre_func: F,
                                          shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<Vec<Timeseries>>
        where F: Fn(&mut [f64], &[i64]) -> () + Send + Sync
    {

        struct Context<'a> {
            func: &'a RollupFunction,
            keep_metric_names: bool,
            iafc: Mutex<IncrementalAggrFuncContext<'a>>,
            rcs: Vec<RollupConfig>,
            timestamps: &'a Arc<Vec<i64>>
        }

        let callbacks = self.incremental_aggr_opts.as_ref().unwrap().callbacks;
        let iafc = Mutex::new(IncrementalAggrFuncContext::new(ae, callbacks));

        let mut ctx = Context {
            keep_metric_names: self.keep_metric_names,
            func: &self.func,
            iafc,
            rcs,
            timestamps: &shared_timestamps
        };

        rss.run_parallel(&mut ctx,  |ctx: Arc<&mut Context>, rs: &mut QueryResult, worker_id: u64| {
            drop_stale_nans(&ctx.func, &mut rs.values, &mut rs.timestamps);
            pre_func(&mut rs.values, &rs.timestamps);

            for rc in ctx.rcs.iter() {
                match new_timeseries_map(&ctx.func, ctx.keep_metric_names, &ctx.timestamps, &rs.metric_name) {
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
                do_rollup_for_timeseries(ctx.keep_metric_names,
                                         rc,
                                         &mut ts,
                                         &rs.metric_name,
                                         &rs.values,
                                         &rs.timestamps,
                                         &ctx.timestamps)?;

                // todo: return result rather than unwrap
                let iafc = ctx.iafc.lock().unwrap();
                iafc.update_timeseries(&mut ts, worker_id).unwrap();
            }
            Ok(())
        })?;

        let mut iafc = ctx.iafc.lock().unwrap();
        Ok(iafc.finalize_timeseries())
    }

    fn eval_no_incremental_aggregate<F>(
        &self,
        rss: &mut QueryResults,
        rcs: Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<Timestamp>>,
        no_stale_markers: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
        where F: Fn(&mut [f64], &[i64]) -> () + Send + Sync
    {

        struct TaskCtx<'a> {
            series: Arc<Mutex<Vec<Timeseries>>>,
            keep_metric_names: bool,
            func: RollupFunction,
            rcs: Vec<RollupConfig>,
            timestamps: &'a Arc<Vec<i64>>,
            no_stale_markers: bool
        }

        let series =  Arc::new(Mutex::new(Vec::with_capacity(rss.len() * rcs.len())));
        let mut ctx = TaskCtx {
            series: Arc::clone(&series),
            keep_metric_names: self.keep_metric_names,
            func: self.func,
            rcs,
            no_stale_markers,
            timestamps: shared_timestamps
        };

        // todo: tinyvec

        rss.run_parallel(&mut ctx,   |ctx: Arc<&mut TaskCtx>, rs: &mut QueryResult, _: u64| {
            if !ctx.no_stale_markers {
                drop_stale_nans(&ctx.func, &mut rs.values, &mut rs.timestamps);
            }
            pre_func(&mut rs.values, &rs.timestamps);
            for rc in ctx.rcs.iter() {
                match new_timeseries_map(&ctx.func, ctx.keep_metric_names, &ctx.timestamps, &rs.metric_name) {
                    Some(tsm) => {
                        rc.do_timeseries_map(&tsm, &rs.values, &rs.timestamps)?;
                        let mut tss = ctx.series.lock().unwrap();
                        tsm.as_ref().borrow_mut().append_timeseries_to( &mut tss);
                    }
                    _ => {
                        let mut ts: Timeseries = Timeseries::default();
                        do_rollup_for_timeseries(
                            ctx.keep_metric_names,
                            rc,
                            &mut ts,
                            &rs.metric_name,
                            &rs.values,
                            &rs.timestamps,
                            ctx.timestamps)?;
                        let mut tss = ctx.series.lock().unwrap();
                        tss.push(ts);
                    }
                }
            }
            Ok(())
        })?;

        // https://users.rust-lang.org/t/how-to-move-the-content-of-mutex-wrapped-by-arc/10259/7
        let res = Arc::try_unwrap(series).unwrap().into_inner().unwrap();

        Ok(res)
    }
}

#[inline]
fn new_timeseries_map(
    func: &RollupFunction,
    keep_metric_names: bool,
    shared_timestamps: &Arc<Vec<Timestamp>>,
    mn: &MetricName) -> Option<Rc<RefCell<TimeseriesMap>>> {
    match TimeseriesMap::new(&func, keep_metric_names, shared_timestamps, &mn) {
        None => None,
        Some(map) => {
           Some(Rc::new(RefCell::new(map)))
        }
    }
}

pub(super) fn compile_rollup_func_args(fe: &FuncExpr) -> RuntimeResult<(Vec<ExprEvaluator>, RollupExpr)> {
    let rollup_arg_idx = fe.get_arg_idx_for_optimization();
    if rollup_arg_idx.is_none() {
        let err = format!("Bug: can't find source arg for rollup function {}. Expr: {}",
                          fe.name, fe);
        return Err(RuntimeError::ArgumentError(err));
    }

    let mut re: RollupExpr = RollupExpr::new(Expression::from(""));

    let arg_idx = rollup_arg_idx.unwrap();

    let mut args: Vec<ExprEvaluator> = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == arg_idx {
            re = get_rollup_expr_arg(arg)?;
            args.push( create_evaluator(&Expression::Rollup(re.clone()))? );
            continue;
        }
        args.push( create_evaluator(&*arg)? );
    }

    return Ok((args, re));
}

// todo: COW. This has a lot of clones
fn get_rollup_expr_arg(arg: &Expression) -> RuntimeResult<RollupExpr> {

    let mut re: RollupExpr = match arg {
        Expression::Rollup(re) => re.clone(),
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
        Expression::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]

            // TODO: avoid clone below. Can we just do an into() ?
            let arg = Expression::Rollup(
                RollupExpr::new(Expression::MetricExpression(me.clone()))
            );

            match FuncExpr::default_rollup(arg) {
                Err(e) => {
                    return Err(
                        RuntimeError::General(format!("{:?}", e))
                    )
                },
                Ok(fe) => {
                    re.expr = Box::new(Expression::Function(fe) );
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


/// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
///
/// Values for returned series are set to nan if at least a single tss series contains nan at that point.
/// This means that tss contains a series with non-empty results at that point.
/// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(ec: &EvalConfig, expr: &Expression, tss: &[Timeseries]) -> Vec<Timeseries> {
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

fn get_keep_metric_names(expr: &Expression) -> bool {
    // todo: move to optimize stage. put result in ast node
    return match expr {
        Expression::Aggregation(ae) => {
            if ae.keep_metric_names {
                return rollup_func_keeps_metric_name(&ae.name)
            }
            false
        }
        Expression::Function(fe) => {
            if fe.keep_metric_names {
                return rollup_func_keeps_metric_name(&fe.name)
            }
            false
        }
        _ => false
    }
}


fn do_parallel<F>(tss: &Vec<Timeseries>, f: F) -> RuntimeResult<Vec<Timeseries>>
    where F: Fn(&Timeseries, &mut [f64], &[i64]) -> RuntimeResult<Vec<Timeseries>> + Send + Sync
{
    let res = tss.par_iter().map(|ts| {
        let len = ts.values.len();
        // todo(perf): use object pool for these
        let mut values: Vec<f64> = Vec::with_capacity(len);
        let mut timestamps: Vec<i64> = Vec::with_capacity(len);

        remove_nan_values(&mut values, &mut timestamps, &ts.values, &ts.timestamps);

        f(ts, &mut values, &mut timestamps)
    }).collect::<Vec<_>>();

    // todo(perf) - figure size of result to preallocate return buf
    let mut series: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for r in res.into_iter() {
        match r {
            Err(e) => return Err(e),
            Ok(timeseries) => series.extend::<Vec<Timeseries>>(timeseries.into())
        }
    }

    return Ok(series)
}

fn remove_nan_values(dst_values: &mut Vec<f64>, dst_timestamps: &mut Vec<i64>, values: &[f64], timestamps: &[i64]) {
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

fn remove_nan_values_in_place(values: &mut Vec<f64>, timestamps: &mut Vec<i64>) {
    if !values.iter().any(|x| x.is_nan()) {
        return;
    }

    // Slow path: drop nans from values.
    let mut k = 0;
    for i in 0 .. values.len() {
        let v = values[i];
        if v.is_nan() {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
    }

    values.truncate(k);
    timestamps.truncate(k);
}

fn do_rollup_for_timeseries(keep_metric_names: bool,
                            rc: &RollupConfig,
                            ts_dst: &mut Timeseries,
                            mn_src: &MetricName,
                            values_src: &[f64],
                            timestamps_src: &[i64],
                            shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<()> {

    ts_dst.metric_name.copy_from(mn_src);
    if rc.tag_value.len() > 0 {
        ts_dst.metric_name.set_tag("rollup", &rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_metric_group();
    }
    rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = Arc::clone(&shared_timestamps);

    Ok(())
}

fn mul_no_overflow(a: i64, b: i64) -> i64 {
    if i64::MAX / b < a {
        // Overflow
        return i64::MAX;
    }
    return a * b;
}

pub(crate) fn drop_stale_nans(func: &RollupFunction, values: &mut Vec<f64>, timestamps: &mut Vec<i64>) {

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
    for i in 0 .. values.len() {
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
