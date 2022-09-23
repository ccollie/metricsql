use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use rayon::iter::IntoParallelRefIterator;
use tinyvec::*;

use lib::is_stale_nan;
use metricsql::ast::*;
use metricsql::functions::{BuiltinFunction, DataType, RollupFunction, Signature, Volatility};

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
use crate::functions::aggregate::IncrementalAggrFuncContext;
use crate::functions::rollup::{eval_prefuncs, get_rollup_configs, get_rollup_function_factory,
                               MAX_SILENCE_INTERVAL, rollup_func_keeps_metric_name, RollupConfig,
                               RollupHandlerEnum, RollupHandlerFactory, TimeseriesMap};
use crate::functions::transform::get_absent_timeseries;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::{join_tag_filterss, QueryResult, QueryResults, SearchQuery};
use crate::timeseries::Timeseries;
use crate::utils::{memory_limit, MemoryLimiter, num_cpus};

use super::traits::Evaluator;

pub(super) struct RollupEvaluator {
    func: RollupFunction,
    signature: Signature,
    pub expr: Expression,
    pub re: RollupExpr,
    pub evaluator: Box<ExprEvaluator>,
    pub args: ArgList,
    pub at: Option<Box<ExprEvaluator>>,
    pub iafc: Option<IncrementalAggrFuncContext>,
    nrf: RollupHandlerFactory,
    keep_metric_names: bool,
}

type SmallFloatArray = TinyVec<[f64; 50]>;

// Init metric
// describe_counter!("vm_rollup_result_cache_full_hits_total", "number of full request cache hits");
// describe_counter!("vm_rollup_result_cache_partial_hits_total", "total number of partial result cache hits");

// static rollupResultCacheFullHits: _ = register_counter!("vm_rollup_result_cache_full_hits_total");
// let rollupResultCachePartialHits = register_counter!("vm_rollup_result_cache_partial_hits_total");
// let rollupResultCacheMiss        = register_counter!("vm_rollup_result_cache_miss_total");

impl Evaluator for RollupEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        self.eval_rollup(ctx, ec)
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }
    
    fn return_type(&self) -> DataType {
        if self.at.is_some() {
            DataType::Series
        } else {
            // todo: check rollup to see
            DataType::Matrix
        }
    }
}

impl RollupEvaluator {
    pub fn new(re: &RollupExpr) -> RuntimeResult<Self> {
        let expr = Expression::Rollup(re.clone());
        let args: Vec<ExprEvaluator> = vec![];
        let mut res = Self::create_internal(RollupFunction::DefaultRollup, re, expr, args)?;
        Ok(res)
    }

    pub fn from_function(expr: &FuncExpr) -> RuntimeResult<Self> {
        let (mut args, re) = compile_rollup_func_args(expr)?;
        // todo: tinyvec

        let func: RollupFunction;
        match expr.function {
            BuiltinFunction::Rollup(rf) => func = rf,
            _ => {
                return Err( RuntimeError::UnknownFunction(
                    format!("Expected a rollup function. Got {}", expr.function.name())
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

        let mut at: Option<Box<ExprEvaluator>>;
        if let Some(re_at) = &re.at {
            let at_expr = create_evaluator(re_at)?;
            at = Some( Box::new(at_expr) );
        } else {
            at = None;
        }

        let keep_metric_names = get_keep_metric_names(&expr);
        let nrf = get_rollup_function_factory(func);
        let evaluator = Box::new(create_evaluator(&expr)?);
        let arg_list = ArgList::from(&signature, args);

        let mut res = RollupEvaluator {
            func,
            signature,
            expr,
            re: re.clone(),
            evaluator,
            args: arg_list,
            at,
            iafc: None,
            nrf,
            keep_metric_names,
        };

        Ok(res)
    }

    // expr may contain:
    // -: RollupFunc(m) if iafc is None
    // - aggrFunc(rollupFunc(m)) if iafc isn't None
    fn eval_rollup(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo: tinyvec
        // todo(perf): if function is not volatile and the non metric args are const, we can store t
        // he results of the nrf call since the result won't change, hence sparing the call to eval
        // the arg list. For ex `quantile(0.95, latency{func="make-widget"})`. We know which is
        let params = self.args.eval(ctx, ec)?;
        let rf = (self.nrf)(&params);

        if self.at.is_none() {
            return self.eval_without_at(ctx, ec, &rf);
        }

        let at_timestamp = self.get_at_timestamp(ctx, ec, &self.at.unwrap())?;
        let mut ec_new = ec.copy_no_timestamps();
        ec_new.start = at_timestamp;
        ec_new.end = at_timestamp;
        let mut tss = self.eval_without_at(ctx, &mut ec_new, &rf)?;

        // expand single-point tss to the original time range.
        let timestamps = ec.timestamps();
        for ts in tss.iter_mut() {
            let v = ts.values[0];
            ts.timestamps = timestamps.clone();
            ts.values = vec![v; timestamps.len()];
        }
        return Ok(tss);
    }

    #[inline]
    fn get_at_timestamp(&self, ctx: &mut Context, ec: &EvalConfig, evaluator: &ExprEvaluator) -> RuntimeResult<i64> {
        match evaluator.eval(ctx, ec) {
            Err(err) => {
                let msg = format!("cannot evaluate `@` modifier");
                return Err(RuntimeError::from(msg));
            }
            Ok(ref tss_at) => {
                if tss_at.len() != 1 {
                    let msg = format!("`@` modifier must return a single series; it returns {} series instead", tss_at.len());
                    return Err(RuntimeError::from(msg));
                }
                Ok((tss_at[0].values[0] * 1000_f64) as i64)
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

    fn eval_without_at(&self, ctx: &mut Context, ec: &EvalConfig, rollup_func: &RollupHandlerEnum) -> RuntimeResult<Vec<Timeseries>> {
        let mut ec_new = ec;

        let mut offset: i64 = self.get_offset(ec.step);
        if offset != 0 {
            let mut with_offset = ec.copy_no_timestamps();
            with_offset.start -= offset;
            with_offset.end -= offset;
            with_offset.ensure_timestamps()?;
            ec_new = &with_offset;

            // There is no need in calling adjust_start_end() on ec_new if ec_new.may_cache is set to true,
            // since the time range alignment has been already performed by the caller,
            // so cache hit rate should be quite good.
            // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
        }

        if self.func == RollupFunction::RollupCandlestick {
            // Automatically apply `offset -step` to `rollup_candlestick` function
            // in order to obtain expected OHLC results.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
            let step = ec_new.step;
            let mut for_candlestick = ec_new.copy_no_timestamps();
            for_candlestick.start += step;
            for_candlestick.end += step;
            offset = offset - step;
            for_candlestick.ensure_timestamps()?;
            ec_new = &for_candlestick;
        }

        let mut rvs = match &*self.re.expr {
            Expression::MetricExpression(me) => {
                self.eval_with_metric_expr(ctx, &ec_new, me, rollup_func)?
            }
            _ => {
                if self.iafc.is_some() {
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
            let mut dst_timestamps: Vec<i64> = Vec::with_capacity(src_timestamps.len());
            for i in 0..dst_timestamps.len() {
                dst_timestamps[i] += offset;
            }
            let shared = Arc::new(dst_timestamps);
            for ts in rvs.iter_mut() {
                ts.timestamps = shared.clone();
            }
        }

        Ok(rvs)
    }

    fn eval_with_subquery(&self,
                          ctx: &mut Context,
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
        if tss_sq.len() == 0 {
            return Ok(vec![]);
        }

        let shared_timestamps = ec.timestamps();
        let (mut rcs, pre_funcs) = get_rollup_configs(
            &self.func,
            rollup_func,
            &self.expr,
            ec.start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            ec.lookback_delta,
            shared_timestamps)?;

        let mut tss_lock: Arc<Mutex<Vec<Timeseries>>> = Arc::new(Mutex::new(
            Vec::with_capacity(tss_sq.len() * rcs.len())
        ));
        let keep_metric_names = self.keep_metric_names;
        let func = self.func;

        do_parallel(&tss_sq, move |ts_sq: &Timeseries, values: &mut [f64], timestamps: &mut [i64]| -> RuntimeResult<()> {

            let len = ts_sq.timestamps.len();

            // Copy vals without NaN
            // Slow path - remove NaNs.
            let mut i = 0;
            for v in ts_sq.values {
                if v.is_nan() {
                    continue;
                }
                i += 1;
            }

            // TODO!!!!!!!!!!!!!

            eval_prefuncs(&pre_funcs, values, timestamps);
            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&func, keep_metric_names, shared_timestamps, &ts_sq.metric_name) {
                    Some(mut tsm) => {
                        rc.do_timeseries_map(&mut tsm, values, timestamps)?;

                        let mut inner = tss_lock.lock().unwrap().deref_mut();
                        tsm.append_timeseries_to(&mut inner);

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
                    shared_timestamps)?;

                let mut inner = tss_lock.lock().unwrap();
                inner.push(ts);
            }

            Ok(())
        })?;

        let lock = Arc::try_unwrap(tss_lock).expect("Lock still has multiple owners");
        let value = lock.into_inner().expect("Mutex cannot be locked");

        Ok(value)
    }

    fn eval_with_incremental_aggregate<F>(&self,
                                       iafc: &mut IncrementalAggrFuncContext,
                                       rss: &mut QueryResults,
                                       rcs: &mut Vec<RollupConfig>,
                                       pre_func: F,
                                       shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<Vec<Timeseries>>
    where F: Fn(&mut [f64], &[i64]) -> () + Send + Sync
    {
        let keep_metric_names = self.keep_metric_names;
        let func = self.func;

        rss.run_parallel(move |rs: &mut QueryResult, worker_id: u64| {
            drop_stale_nans(&func, &mut rs.values, &mut rs.timestamps);
            pre_func(&mut rs.values, &rs.timestamps);
            let mut ts = get_timeseries();

            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&func, keep_metric_names, shared_timestamps, &rs.metric_name) {
                    Some(mut tsm) => {
                        rc.do_timeseries_map(&mut tsm, &rs.values, &rs.timestamps)?;
                        for mut ts in tsm.values_mut() {
                            iafc.update_timeseries(&mut ts, worker_id)?;
                        }
                        continue;
                    }
                    _ => {}
                }

                ts.reset();
                do_rollup_for_timeseries(keep_metric_names,
                                         rc,
                                         &mut ts,
                                         &rs.metric_name,
                                         &rs.values,
                                         &rs.timestamps,
                                         &shared_timestamps)?;

                // todo: return result rather than unwrap
                iafc.update_timeseries(&mut ts, worker_id).unwrap();
            }
            Ok(())
        })?;

        Ok(iafc.finalize_timeseries())
    }

    fn eval_with_metric_expr(
        &self,
        ctx: &mut Context,
        ec: &EvalConfig,
        me: &MetricExpr,
        rollup_func: &RollupHandlerEnum,
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = self.get_window(ec.step);

        if me.is_empty() {
            return Ok(eval_number(ec, f64::NAN));
        }

        // Search for partial results in cache.
        let (tss_cached, start) = ctx.rollup_result_cache.get(ec, &self.expr, window)?;
        let tss_cached = tss_cached.unwrap();

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

        let (mut rcs, pre_funcs) = get_rollup_configs(
            &self.func,
            rollup_func,
            &self.expr,
            start,
            ec.end,
            ec.step,
            window,
            ec.max_points_per_series,
            ec.lookback_delta,
            &shared_timestamps)?;

        let pre_func = move |values, timestamps| {
            eval_prefuncs(&pre_funcs, values, timestamps)
        };

        // Fetch the remaining part of the result.
        let tfs = vec![me.label_filters];
        let tfss = join_tag_filterss(&tfs, &ec.enforced_tag_filterss);
        let mut min_timestamp = start - MAX_SILENCE_INTERVAL;
        if window > ec.step {
            min_timestamp -= &window
        } else {
            min_timestamp -= ec.step
        }
        let filters = tfss.to_vec();
        let sq = SearchQuery::new(min_timestamp, ec.end, filters, ec.max_series);
        let mut rss = ctx.series_data.search(&sq, &ec.deadline)?;
        let rss_len = rss.len();
        if rss_len == 0 {
            rss.cancel();
            let mut dst: Vec<Timeseries> = Vec::with_capacity(tss_cached.len());
            let tss = merge_timeseries(&tss_cached, &mut dst, start, ec)?;
            return Ok(tss);
        }

        // Verify timeseries fit available memory after the rollup.
        // Take into account points from tss_cached.
        let points_per_timeseries = 1 + (ec.end - ec.start) / ec.step;
        let mut timeseries_len = rss_len;

        let iafc: &IncrementalAggrFuncContext;

        match &self.iafc {
            Some(ctx) => {
                iafc = ctx;
                // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
                timeseries_len = usize::from(num_cpus()?);
                if iafc.modifier.is_some() {
                    if iafc.limit > 0 {
                        // There is an explicit limit on the number of output time series.
                        timeseries_len *= iafc.limit
                    } else {
                        // Increase the number of timeseries for non-empty group list: `aggr() by (something)`,
                        // since each group can have own set of time series in memory.
                        timeseries_len *= 1000
                    }
                }
                // The maximum number of output time series is limited by rss_len.
                if timeseries_len > rss_len {
                    timeseries_len = rss_len
                }
            }
            _ => {}
        }

        let rollup_points = mul_no_overflow(points_per_timeseries, (timeseries_len * rcs.len()) as i64);
        let rollup_memory_size = mul_no_overflow(rollup_points, 16) as usize;

        if !ctx.rollup_result_cache.reserve_memory(rollup_memory_size) {
            rss.cancel();
            let msg = format!("not enough memory for processing {} data points across {} time series with {} points in each time series; \n
                                  total available memory for concurrent requests: {} bytes; requested memory: {} bytes; \n
                                  possible solutions are: reducing the number of matching time series; switching to node with more RAM; \n
                                  increasing -memory.allowedPercent; increasing `step` query arg ({})",
                              rollup_points,
                              timeseries_len * rcs.len(),
                              points_per_timeseries,
                              ctx.rollup_result_cache.memory_limiter.max_size,
                              rollup_memory_size as u64,
                              ec.step as f64 / 1e3
            );

            return Err(RuntimeError::from(msg));
        }

        defer! {
            ctx.rollup_result_cache.release_memory(rollup_memory_size);
        }

        // Evaluate rollup
        // shadow timestamps
        let shared_timestamps = Arc::new(shared_timestamps);
        let mut tss: &Vec<Timeseries>;
        if let Some(mut iafc) = &self.iafc {
            tss = &self.eval_with_incremental_aggregate(
                &mut iafc,
                &mut rss,
                &mut rcs,
                &pre_func,
                &shared_timestamps)?;
        } else {
            tss = &self.eval_no_incremental_aggregate(
                &mut rss,
                &mut rcs,
                &pre_func,
                &shared_timestamps,
                ec.options.no_stale_markers
            )?;
        }

        let res = merge_timeseries(&tss_cached, &mut tss, start, ec)?;
        ctx.rollup_result_cache.put(ec, &self.expr, window, &res)?;


        Ok(res)
    }

    fn eval_no_incremental_aggregate<F>(
        &self,
        rss: &mut QueryResults,
        rcs: &mut Vec<RollupConfig>,
        pre_func: F,
        shared_timestamps: &Arc<Vec<i64>>,
        no_stale_markers: bool,
    ) -> RuntimeResult<Vec<Timeseries>>
        where F: Fn(&mut [f64], &[i64]) -> () + Send + Sync
    {

        let tss_lock: Arc<Mutex<Vec<Timeseries>>> = Arc::new(Mutex::new(
            Vec::with_capacity(rss.len() * rcs.len())
        ));

        let keep_metric_names = self.keep_metric_names;

        // todo: tinyvec
        let func = self.func;

        rss.run_parallel(move |rs: &mut QueryResult, worker_id: u64| {
            if !no_stale_markers {
                drop_stale_nans(&func, &mut rs.values, &mut rs.timestamps);
            }
            pre_func(&mut rs.values, &rs.timestamps);
            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&func, keep_metric_names, shared_timestamps, &rs.metric_name) {
                    Some(mut tsm) => {
                        rc.do_timeseries_map(&mut tsm, &rs.values, &rs.timestamps)?;
                        let mut _tss = tss_lock.lock().unwrap().deref_mut();
                        tsm.append_timeseries_to( _tss);
                    }
                    _ => {
                        let mut ts: Timeseries = Timeseries::default();
                        do_rollup_for_timeseries(
                            keep_metric_names,
                            rc,
                            &mut ts,
                            &rs.metric_name,
                            &rs.values,
                            &rs.timestamps,
                            shared_timestamps)?;
                        let mut _tss = tss_lock.lock().unwrap();
                        _tss.push(ts);
                    }
                }
            }
            Ok(())
        })?;

        // https://users.rust-lang.org/t/how-to-move-the-content-of-mutex-wrapped-by-arc/10259/7
        let res = Arc::try_unwrap(tss_lock).unwrap().into_inner().unwrap();

        Ok(res)
    }
}

pub(super) fn compile_rollup_func_args(fe: &FuncExpr) -> RuntimeResult<(Vec<ExprEvaluator>, &RollupExpr)> {
    let rollup_arg_idx = fe.get_arg_idx_for_optimization();
    if rollup_arg_idx.is_none() {
        let err = format!("Bug: can't find source arg for rollup function {}. Expr: {}",
                          fe.name(), fe);
        return Err(RuntimeError::ArgumentError(err));
    }

    let mut re: RollupExpr = RollupExpr {
        expr: Box::new(Expression::from("")),
        window: None,
        offset: None,
        step: None,
        inherit_step: false,
        at: None,
        span: None
    };

    let arg_idx = rollup_arg_idx.unwrap();

    let mut args: Vec<ExprEvaluator> = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == arg_idx {
            re = get_rollup_expr_arg(arg)?;
            args[i] = create_evaluator(&Expression::Rollup(re))?;
            continue;
        }
        args[i] = create_evaluator(&*arg)?;
    }

    return Ok((args, &re));
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
                Ok(mut fe) => {
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
                return rollup_func_keeps_metric_name(&fe.name())
            }
            false
        }
        _ => false
    }
}

fn do_parallel<F>(tss: &Vec<Timeseries>, f: F) -> RuntimeResult<()>
where F: Fn(&Timeseries, &mut [f64], &mut [i64]) -> RuntimeResult<()>
{
    let mut tmp_values: TinyVec<[f64; 32]> = tiny_vec!();
    let mut tmp_timestamps = tiny_vec!([i64; 32]);

    let mut err: Option<RuntimeError> = None;
    tss.iter().par_iter().for_each(|ts| {
        match f(ts, tmp_values.as_mut(), tmp_timestamps.as_mut()) {
            Err(e) => {
                if err.is_none() {
                    err = Some(e);
                }
            }
            _ => {}
        }
    });

    return match err {
        Some(e) => Err(e),
        None => Ok(())
    };
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

pub(crate) static ROLLUP_MEMORY_LIMITER: Lazy<MemoryLimiter> = Lazy::new(|| {
    MemoryLimiter::new((memory_limit().unwrap() / 4) as usize) // todo: move calc out
});


fn do_rollup_for_timeseries(keep_metric_names: bool,
                            rc: &mut RollupConfig,
                            ts_dst: &mut Timeseries,
                            mn_src: &MetricName,
                            values_src: &[f64],
                            timestamps_src: &[i64],
                            shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<()> {
    ts_dst.metric_name.copy_from(mn_src);
    if rc.tag_value.len() > 0 {
        ts_dst.metric_name.add_tag("rollup", &rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_metric_group();
    }
    rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = shared_timestamps.clone();

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
    let mut dst_values = values;

    let mut k = 0;
    let mut i = 0;
    for v in values.iter() {
        if !is_stale_nan(*v) {
            dst_values[k] = *v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
        i += 1
    }

    dst_values.truncate(k);
    timestamps.truncate(k);
}
