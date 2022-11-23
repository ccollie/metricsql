use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

use once_cell::sync::{Lazy, OnceCell};
use rayon::iter::IntoParallelRefIterator;
use regex::internal::Input;
use regex::Regex;

use metricsql::{get_rollup_arg_idx, pushdown_binary_op_filters, trim_filters_by_group_modifier};
use metricsql::types::*;

use crate::{EvalConfig, get_timeseries, get_timestamps, MetricName, TimeseriesMap};
use crate::aggr::{AggrFuncArg, get_aggr_func};
use crate::aggr_incremental::{get_incremental_aggr_func_callbacks, IncrementalAggrFuncContext};
use crate::cache::merge_timeseries;
use crate::eval::{copy_eval_config, create_evaluator, eval_number, validate_max_points_per_timeseries};
use crate::eval::aggregate::try_get_arg_rollup_func_with_metric_expr;
use crate::eval::timeseries_map::TimeseriesMap;
use crate::eval::traits::{EmptyEvaluator, Evaluator};
use crate::rollup::{
    get_rollup_configs,
    get_rollup_func,
    NewRollupFunc,
    rollup_func_keeps_metric_name,
    RollupArgValue,
    RollupConfig,
    RollupFunc,
};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::tag_filter::{TagFilter, to_tag_filters};
use crate::timeseries::Timeseries;
use crate::transform::get_absent_timeseries;

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct RollupEvaluator {
    func_name: String,
    expr: Expression,
    re: RollupExpr,
    iafc: Option<IncrementalAggrFuncContext>,
    nrf: &'static NewRollupFunc,
    pub(super)_evaluator: Box<dyn Evaluator>,
    pub(super) rollup_index: i32,
    pub(super) args: Vec<Box<dyn Evaluator>>,
    at: Option<Box<dyn Evaluator>>,
    keep_metric_names: bool,
}

impl Evaluator for RollupEvaluator {
    fn eval<'a>(&self, ec: &'a mut EvalConfig) -> RuntimeResult<&'a Vec<Timeseries>> {
        let res = self.eval_rollup(ec)?;
        Ok(&res)
    }
}

impl RollupEvaluator {
    pub(crate) fn new(re: &RollupExpr) -> Self {
        let expr = Expression::cast(re);
        let args: Vec<Box<dyn Evaluator>> = vec![];
        let mut res = Self::create_internal("default_rollup", re, &expr, args);
        res.evaluator = create_evaluator(&expr)?;
        res
    }

    pub(crate) fn from_function(expr: &FunctionExpr) -> Self {
        let (mut args, re, rollup_index) = compile_rollup_func_args(fe)?;
        // todo: tinyvec

        // see docs fpr Vec::swap_remove
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove
        args.push(Box::new(EmptyEvaluator{}) );
        let evaluator = args.swap_remove(rollup_index);

        let mut res = Self::create_internal(func_name, &re, expr, args);
        res.evaluator = evaluator;
        res.args = args.into();

        res
    }

    pub(self) fn from_metric_expression(me: &MetricExpression) -> Self {
        let re = RollupExpr::wrap(me);
        self::new(re)
    }

    pub(super) fn create_internal(name: &str, re: &RollupExpr, expr: &Expression, args: Vec<Box<dyn Evaluator>>) -> Self {
        let mut res = RollupEvaluator::default();
        res.func_name = name.to_lowercase().to_string();
        res.re = re.clone();
        res.args = args;

        if let Some(func) = get_rollup_func(func_name) {
            res.nrf = func;
        } else {
            panic!("BUG: Unknown rollup function {}", res.func_name);
        }

        if let Some(re_at) = expr.at {
            let at_eval = create_evaluator(re_at)?;
            res.at = Some(at_eval);
            res.keep_metric_names = get_keep_metric_names(expr);
        }

        res
    }

    // expr may contain:
    // -: RollupFunc(m) if iafc is nil
    // - aggrFunc(rollupFunc(m)) if iafc isn't nil
    fn eval_rollup(&self, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo: tinyvec
        let params: Vec<Vec<Timeseries>> = Vec::with_capacity(self.args.len());
        for (i, arg) in self.args.iter().enumerate() {
            if i == self.rollup_index {
                params[i] = vec![]
            } else {
                params[i] = *arg.eval(ec);
            }
        }

        if self.at.is_none() {
            return self.eval_without_at(ec, rf);
        }

        let mut tss_at: &Vec<Timeseries>;

        match self.at.unwrap().eval(ec) {
            Err(err) => {
                let msg = format!("cannot evaluate `@` modifier: {}", err);
                return Err(RuntimeError::from(msg));
            }
            Ok(res) => tss_at = res
        };

        if tss_at.len() != 1 {
            let msg = format!("`@` modifier must return a single series; it returns {} series instead", tss_at.len());
            return Err(RuntimeError::from(msg));
        }

        let at_timestamp = tss_at[0].values[0] * 1000 as i64;
        let mut ec_new = ec.copy_no_timestamps();
        ec_new.start = at_timestamp;
        ec_new.end = at_timestamp;
        let mut tss = self.eval_without_at(&mut ec_new, rf)?;

        // expand single-point tss to the original time range.
        let timestamps = ec.get_shared_timestamps();
        let rc_timestamps = Rc::new(timestamps);
        for ts in tss.iter_mut() {
            let v = ts.values[0];
            ts.timestamps = rc_timestamps.clone();
            ts.values = vec![v; timestamps.len()];
        }
        return Ok(tss);
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

    fn eval_without_at(&self, ec: &mut EvalConfig, rollup_func: &RollupFunc) -> RuntimeResult<Vec<Timeseries>> {
        let mut ec_new = ec;
        let mut offset: i64 = self.get_offset(ec.step);
        if offset != 0 {
            ec_new = &mut ec_new.copy_no_timestamps();
            ec_new.start = ec_new.start - offset;
            ec_new.end = ec_new.end - offset;
            // There is no need in calling adjust_start_end() on ec_new if ec_new.may_cache is set to true,
            // since the time range alignment has been already performed by the caller,
            // so cache hit rate should be quite good.
            // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
        }

        if self.f_name == "rollup_candlestick" {
            // Automatically apply `offset -step` to `rollup_candlestick` function
            // in order to obtain expected OHLC results.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
            let step = ec_new.step;
            ec_new = &mut ec_new.copy_no_timestamps();
            ec_new.start = ec_new.start + step;
            ec_new.end = ec_new.end + step;
            offset = offset - step;
        }

        let mut rvs = match &self.re.expr {
            Expression::MetricExpression(me) => {
                self.eval_with_metric_expr(&mut ec_new, me, rollup_func)?
            }
            _ => {
                if Some(iafc) {
                    let msg = format!("BUG: iafc must be null for rollup {} over subquery {}", func_name, re);
                    return Err(RuntimeError::from(msg));
                }
                self.eval_with_subquery(&mut ec_new, rollup_func)?
            }
        };

        if self.func_name == "absent_over_time" {
            rvs = aggregate_absent_over_time(&ec, &re.expr, &rvs)
        }

        if offset != 0 && rvs.len() > 0 {
            // Make a copy of timestamps, since they may be used in other values.
            let src_timestamps = rvs[0].timestamps;
            let mut dst_timestamps: Vec<i64> = Vec::with_capacity(src_timestamps.len());
            for i in 0..dst_timestamps.len() {
                dst_timestamps[i] += offset;
            }
            let shared = Rc::new(dst_timestamps);
            for ts in rvs {
                ts.timestamps = shared.clone();
            }
        }
        Ok(rvs)
    }

    fn eval_with_subquery(&self, ec: &mut EvalConfig, rollup_func: &RollupFunc) -> RuntimeResult<Vec<Timeseries>> {
        // TODO: determine whether to use rollupResultCacheV here.

        let step = self.get_step(ec.step);
        let window = self.get_window(ec.step);

        let mut ec_sq = ec.copy_no_timestamps();
        ec_sq.start -= window + maxSilenceInterval + step;
        ec_sq.end += step;
        ec_sq.step = step;
        validate_max_points_per_timeseries(ec_sq.start, ec_sq.end, ec_sq.step)?;

        // unconditionally align start and end args to step for subquery as Prometheus does.
        (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);
        let tss_sq = self.evaluator.eval(&mut ec_sq)?;
        if tss_sq.len() == 0 {
            return Ok(vec![]);
        }

        let shared_timestamps = ec.get_shared_timestamps();
        let (pre_func, mut rcs) = get_rollup_configs(
            &self.func_name,
            rollup_func,
            &self.expr,
            ec.start,
            ec.end,
            ec.step,
            window,
            ec.lookback_delta,
            shared_timestamps)?;

        let mut tss: Vec<Timeseries> = Vec::with_capacity(tss_sq.len() * rcs.len());
        let mut tss_lock: Arc<RwLock<Vec<Timeseries>>> = Arc::new(RwLock::new(tss));
        let keep_metric_names = self.keep_metric_names;

        do_parallel(&tss_sq, |tsSQ: Timeseries, values: &mut Vec<f64>, timestamps: &mut Vec<i64>| -> (Vec<f64>, Vec<i64>) {
            remove_nan_values(values, timestamps, &tsSQ.values, &tsSQ.timestamps);
            pre_func(values, timestamps);
            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&self.func_name, keep_metric_names, shared_timestamps, &tsSQ.metric_name) {
                    Some(mut tsm) => {
                        rc.do_timeseries_map(_tsm, values, timestamps);

                        let mut inner = tss_lock.lock().unwrap().deref();
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
                    &tsSQ.metric_name,
                    &values,
                    &timestamps,
                    shared_timestamps);

                let mut inner = tss_lock.lock().unwrap();
                inner.push(ts);
            }
            return (values, timestamps);
        });
        return tss;
    }

    fn eval_with_incremental_aggregate(&self,
                                       iafc: &mut IncrementalAggrFuncContext,
                                       rss: &QueryResults,
                                       rcs: &mut [RollupConfig],
                                       pre_func: &PreFunc,
                                       shared_timestamps: &[i64]) -> RuntimeResult<Vec<Timeseries>> {
        let func_name = self.func_name.clone();

        rss.run_parallel(|mut rs: &QueryResult, worker_id: u64| {
            drop_stale_nans(&func_name, rs.values, rs.timestamps);
            pre_func(&mut rs.values, rs.timestamps);
            let mut ts = get_timeseries();

            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&func_name, keep_metric_names, shared_timestamps, &rs.metric_name) {
                    Some(tsm) => {
                        rc.do_timeseries_map(&tsm, rs.values, rs.timestamps);
                        for ts in tsm.m.values() {
                            iafc.update_timeseries(ts, workerID)
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
                                         rs.values,
                                         rs.timestamps,
                                         &shared_timestamps);

                iafc.update_timeseries(&ts, workerID);
            }
        });

        let tss = iafc.finalize_timeseries();
        return tss;
    }

    fn eval_with_metric_expr(
        &self,
        ec: &mut EvalConfig,
        me: &MetricExpr,
        rollup_func: &RollupFunc
    ) -> RuntimeResult<Vec<Timeseries>> {
        let window = self.get_window(ec.step);

        if me.is_empty() {
            return Ok(eval_number(ec, nan));
        }

        // Search for partial results in cache.
        let (tss_cached, start) = rollupResultCacheV.get(ec, expr, &window);
        if start > ec.end {
            // The result is fully cached.
            rollupResultCacheFullHits.inc();
            return tss_cached;
        }

        if start > ec.start {
            rollupResultCachePartialHits.inc();
        } else {
            rollupResultCacheMiss.Inc();
        }

        // Obtain rollup configs before fetching data from db,
        // so type errors can be caught earlier.
        let shared_timestamps = get_timestamps(start, ec.end, ec.step)?;
        let (pre_func, mut rcs) = get_rollup_configs(
            &self.func_name,
            rollup_func,
            expr,
            start,
            ec.end,
            ec.step,
            window,
            ec.lookback_delta,
            &shared_timestamps)?;

        // Fetch the remaining part of the result.
        let tfs = to_tag_filters(&me.label_filters);
        let tfss = join_tag_filterss(vec![vec![tfs]], &ec.enforced_tag_filterss);
        let min_timestamp = start - maxSilenceInterval;
        if window > ec.step {
            min_timestamp -= &window
        } else {
            min_timestamp -= ec.step
        }
        let sq = SearchQuery::new(min_timestamp, ec.end, tfss, ec.max_series);
        let rss = netstorage.ProcessSearchQuery(sq, true, ec.deadline)?;
        let rss_len = rss.len();
        if rss_len == 0 {
            rss.cancel();
            let mut dst: Vec<Timeseries> = Vec::with_capacity(tss_cached.len());
            tss = merge_timeseries(tss_cached, &mut dst, start, ec);
            return tss;
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
                timeseries_len = num_cpus();
                if iafc.ae.modifier.op != "" {
                    if iafc.ae.limit > 0 {
                        // There is an explicit limit on the number of output time series.
                        timeseries_len *= iafc.ae.Limit
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

        let rollup_points = mul_no_overflow(points_per_timeseries, timeseries_len * rcs.len());
        let rollup_memory_size = mul_no_overflow(rollup_points, 16);
        let rml = getRollupMemoryLimiter();
        if !rml.get(rollup_memory_size) {
            rss.cancel();
            let msg = format!("not enough memory for processing {} data points across {} time series with {} points in each time series; " +
                                  "total available memory for concurrent requests: {} bytes; " +
                                  "requested memory: {} bytes; " +
                                  "possible solutions are: reducing the number of matching time series; switching to node with more RAM; " +
                                  "increasing -memory.allowedPercent; increasing `step` query arg ({})",
                              rollup_points,
                              timeseries_len * rcs.len(),
                              points_per_timeseries,
                              rml.max_size,
                              rollup_memory_size as u64,
                              ec.step as f64 / 1e3
            );

            return Err(RuntimeError::from(msg));
        }

        defer! {
            rml.put(rollup_memory_size);
        }

        // Evaluate rollup
        let keep_metric_names = get_keep_metric_names(expr);
        let mut tss: &Vec<Timeseries>;
        if iafc.is_some() {
            let mut iafc = iafc.unwrap();
            tss = eval_with_incremental_aggregate(
                &func_name,
                keep_metric_names,
                &mut iafc,
                rss,
                rcs,
                pre_func,
                &shared_timestamps)?;
        } else {
            tss = &self.eval_no_incremental_aggregate(
                rss,
                &mut rcs,
                pre_func,
                &shared_timestamps)?;
        }

        let tss = merge_timeseries(tss_cached, &mut tss, start, ec);
        rollupResultCacheV.put(qt, ec, expr, window, tss);
        return tss;
    }

    fn eval_no_incremental_aggregate(
        &self,
        rss: &QueryResults,
        rcs: &mut Vec<RollupConfig>,
        pre_func: PreFunc,
        shared_timestamps: &[i64],
    ) -> RuntimeResult<Vec<Timeseries>> {
        let tss: Vec<Timeseries> = Vec::with_capacity(rss.len() * rcs.len());

        let tss_lock: Arc<Mutex<Vec<Timeseries>>> = Arc::new(Mutex::new(tss));

        let keep_metric_names = self.keep_metric_names;

        // todo: tinyvec
        let func_name = self.func_name.clone();

        rss.run_parallel(move |mut rs: &QueryResult| {
            drop_stale_nans(&func_name, rs.values, rs.timestamps);
            pre_func(rs.values, rs.timestamps);
            for rc in rcs.iter_mut() {
                match TimeseriesMap::new(&func_name, keep_metric_names, shared_timestamps, &rs.metric_name) {
                    Some(mut tsm) => {
                        rc.do_timeseries_map(tsm, rs.values, rs.timestamps);
                        let _tss = tss_lock.unlock().unwrap();
                        tsm.append_timeseries_to(_tss);
                    }
                    _ => {
                        let mut ts: Timeseries = Timeseries::default();
                        do_rollup_for_timeseries(
                            keep_metric_names,
                            rc,
                            &mut ts,
                            &rs.metric_name,
                            rs.values,
                            rs.timestamps,
                            shared_timestamps);
                        let _tss = tss_lock.unlock().unwrap();
                        _tss.push(ts);
                    }
                }
            }
        });

        return tss;
    }
}

///////////////////////////////////////

pub(super) fn compile_rollup_func_args(fe: &FuncExpr) -> RuntimeResult<(Vec<Box<dyn Evaluator>>, RollupExpr, usize)> {
    // todo: can this check bbe done during parsing ?
    let rollup_arg_idx = get_rollup_arg_idx(fe);
    if fe.args.len() <= rollup_arg_idx as usize {
        let err = format!("expecting at least {} args to {}; got {} args; expr: {}",
                          rollup_arg_idx + 1, fe.name, re.args.len(), fe);
        return Err(RuntimeError::InvalidArgument(err));
    }

    let mut init = false;
    let mut re: RollupExpr;
    let mut args: Vec<Box<dyn Evaluator>> = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg);
            args[i] = create_evaluator(&Expression::cast(&re))?;
            init = true;
            continue;
        }
        args[i] = create_evaluator(&*arg)?;
    }
    // this is here only to quiet the compiler
    if !init {
        re = RollupExpr::default();
    }
    return Ok((args, re, rollup_arg_idx as usize));
}

fn get_rollup_expr_arg(arg: &Expression) -> RollupExpr {
    let mut re: RollupExpr = match arg {
        Expression::Rollup(re) => re.clone(),
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            RollupExpr::wrap(arg)
        }
    };
    if !re.for_subquery() {
        // Return standard rollup if it doesn't contain subquery.
        return re;
    }
    return match re.expr {
        Expression::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]
            let mut re_new = re.clone();
            let rollup = RollupExpr::wrap(me);
            let expr = Expression::cast(rollup);
            re_new.expr = Box::new(FuncExpr::from_single_arg("default_rollup", expr));
            &re_new
        }
        _ => {
            // arg contains subquery.
            re
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
                rvs[0].values[i] = nan;
                break;
            }
        }
    }
    return rvs;
}

fn get_keep_metric_names(expr: &Expression) -> bool {
    // todo: move to optimize stage. put result in ast node
    match expr {
        Expression::Aggregation(ae) => {
            *ae.keep_metric_names && rollup_func_keeps_metric_name(ae.func_name)
        }
        Expression::Function(fe) => {
            *fe.keep_metric_names && rollup_func_keeps_metric_name(fe.func_name)
        }
        _ => false
    }
}

fn do_parallel(tss: &Vec<Timeseries>, f: fn(ts: &Timeseries, values: &[f64], timestamps: &[i64]) -> (Vec<f64>, Vec<i64>)) {
    let mut tmp_values: Vec<f64> = Vec::with_capacity(1);
    let mut tmp_timestamps: Vec<i64> = Vec::with_capacity(1);

    tss.par_iter().for_each(|x| {
        (tmp_values, tmp_timestamps) = f(ts, &tmp_values, &tmp_timestamps);
    });

    (tmp_values, tmp_timestamps)
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

static ROLLUP_MEMORY_LIMITER: Lazy<MemoryLimiter> = Lazy::new(|| {
    MemoryLimiter::new(memory::memory_limit() / 4) // todo: move calc out
});


fn do_rollup_for_timeseries(keep_metric_names: bool,
                            rc: &mut RollupConfig,
                            ts_dst: &mut Timeseries,
                            mn_src: &MetricName,
                            values_src: &Vec<f64>,
                            timestamps_src: &Vec<i64>,
                            shared_timestamps: &[i64]) {
    ts_dst.metric_name.copy_from(mn_src);
    if rc.tag_value.len() > 0 {
        ts_dst.metric_name.add_tag("rollup", &rc.tag_value)
    }
    if !keep_metric_names {
        ts_dst.metric_name.reset_metric_group();
    }
    rc.exec(&mut ts_dst.values, values_src, timestamps_src)?;
    ts_dst.timestamps = shared_timestamps;
    ts_dst.denyReuse = true
}

fn mul_no_overflow(a: i64, b: i64) -> i64 {
    if i64::MAX / b < a {
        // Overflow
        return i64::MAX;
    }
    return a * b;
}

pub(crate) fn drop_stale_nans(func_name: &str, values: &mut Vec<f64>, timestamps: &mut Vec<i64>) {
    if *noStaleMarkers || func_name == "default_rollup" || func_name == "stale_samples_over_time" {
        // do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return ();
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = values.iter().any(|x| is_stale_nan(x));

    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return ();
    }
    // Slow path: drop Prometheus staleness marks from values.
    let mut dst_values = values;

    let mut k = 0;
    let mut i = 0;
    for v in values.iter() {
        if !is_stale_nan(v) {
            *dst_values[k] = *v;
            *timestamps[k] = timestamps[i];
            k = k + 1;
        }
        i = i + 1
    }

    dst_values.truncate(k);
    timestamps.truncate(k);
}
