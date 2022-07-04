use std::collections::{BTreeMap, BTreeSet, Vec};
use regex::Regex;
use lib::error::{Error, Result};
use metricsql::types::*;
use crate::aggr_incremental::IncrementalAggrFuncContext;
use crate::rollup::{get_rollup_func, NewRollupFunc, RollupFunc};
use super::binary_op::*;
use std::sync::{Arc, Mutex};

// The minimum number of points per timeseries for enabling time rounding.
// This improves cache hit ratio for frequently requested queries over
// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: usize = 50;

// let mut disableCache = false;

use crate::timeseries::Timeseries;
use crate::transform::TransformFuncArg;

pub fn adjust_start_end(start: i64, end: i64, step: i64) -> (i64, i64) {
    if disableCache {
        // Do not adjust start and end values when cache is disabled.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/563
        return (start, end);
    }
    let points = (end - start) / step + 1;
    if points < MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING {
        // Too small number of points for rounding.
        return (start, end);
    }

    // Round start and end to values divisible by step in order
    // to enable response caching (see EvalConfig.mayCache).
    let (start, end) = align_start_end(start, end, step);

    // Make sure that the new number of points is the same as the initial number of points.
    let mut new_points = (end - start) / step + 1;
    let mut _end = end;
    while new_points > points {
        _end = end - step;
        new_points = new_points - 1;
    }

    return (start, _end);
}

fn align_start_end(start: &i64, end: &i64, step: &i64) -> (i64, i64) {
    // Round start to the nearest smaller value divisible by step.
    let new_start = start - start % step;
    // Round end to the nearest bigger value divisible by step.
    let adjust = end % step;
    let mut mew_end = end;
    if adjust > 0 {
        new_end += step - adjust
    }
    return (new_start, new_end);
}

#[derive(Display, Copy, Clone, Debug)]
pub struct EvalConfig {
    pub start: i64,
    pub end: i64,
    pub step: i64,

    // max_series is the maximum number of time series, which can be scanned by the query.
    // Zero means 'no limit'
    pub max_series: usize,

    // QuotedRemoteAddr contains quoted remote address.
    pub quoted_remote_addr: Option<String>,

    // Deadline searchutils.Deadline

    // Whether the response can be cached.
    _mayCache: bool,

    // LookbackDelta is analog to `-query.lookback-delta` from Prometheus.
    pub lookback_delta: i64,

    // How many decimal digits after the point to leave in response.
    pub round_digits: i16,

    // EnforcedTagFilterss may contain additional label filters to use in the query.
    // EnforcedTagFilterss [][]TagFilter

    pub timestamps: Vec<i64>,
    // timestampsOnce sync.Once
}

impl EvalConfig {
    pub fn new() -> Self {
        EvalConfig {
            start: 0,
            end: 0,
            step: 0,
            max_series: 0,
            quoted_remote_addr: None,
            _mayCache: false,
            lookback_delta: 0,
            round_digits: 0,
            timestamps: vec![],
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.start > self.end {
            let msg = format!("BUG: start cannot exceed end; got {} vs {}", ec.Start, ec.End);
            return Err(Error::new(msg));
        }
        if self.step <= 0 {
            let msg = format!("BUG: step must be greater than 0; got {}", ec.Step);
            return Err(Error::new(msg));
        }
        Ok(())
    }

    pub fn may_cache(&self) -> bool {
        if disableCache {
            return false;
        }
        if self._mayCache {
            return true;
        }
        if self.start % self.step != 0 {
            return false;
        }
        if self.end % self.step != 0 {
            return false;
        }
        return true;
    }
}

fn get_timestamps(start: &i64, end: &i64, step: &i64) -> Result<Vec<i64>> {
    // Sanity checks.
    if step <= 0 {
        let msg = format!("BUG: Step must be bigger than 0; got {}", step);
        return Err(Error::new(msg));
    }
    if start > end {
        let msg = format!("BUG: Start cannot exceed End; got {} vs {}", start, end);
        return Err(Error::new(msg));
    }
    match validateMaxPointsPerTimeseries(start, end, step) {
        Err(err) => {
            let msg = format!("BUG: {}; this must be validated before the call to get_timestamps", err);
            return Err(Error::new(msg));
        }
        _ => ()
    }

    // Prepare timestamps.
    let n = 1 + (end - start) / step;
    let mut timestamps: Vec<i64> = Vec::create_with_capacity(n);
    let mut cursor = start;
    for i in 0..n {
        timestamps.push(*cursor);
        cursor = &(cursor + step)
    }
    return Ok(timestamps);
}

pub fn eval_expr(qt: &QueryTracer, ec: &EvalConfig, e: &Expression) -> Result<Vec<Timeseries>> {
    if qt.enabled() {
        let may_cache = ec.mayCache();
        qt = qt.new_child("eval: query={}, timeRange=[{}..{}}], step={}, may_cache={}", e, ec.start, ec.end, ec.step, may_cache)
    }
    let rv = eval_expr_internal(qt, ec, e)?;
    if qt.enabled() {
        seriesCount = rv.len();
        let mut points_per_series = 0;
        if rv.len() > 0 {
            points_per_series = rv[0].timestamps.len();
        }
        let points_count = seriesCount * points_per_series;
        qt.Donef("series={}, points={}, points_per_series={}", seriesCount, points_count, points_per_series)
    }
    Ok(rv)
}

pub fn eval_expr_internal(qt: &Querytracer, ec: &EvalConfig, e: &MetricExpr) -> Result<Vec<Timeseries>> {
    match e {
        Expression::MetricExpression(me) => {
            let re = RollupExpr::new(me);
            match eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, None) {
                Err(err) => {
                    fmt.Errorf("cannot evaluate {}: {}", me, err)
                }
                Ok(res) => res
            }
        }
        Expression::Rollup(re) => {
            match eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, None) {
                Err(err) => {
                    fmt.Errorf("cannot evaluate {}: {}", me, err)
                }
                Ok(res) => res
            }
        }
        Expression::Function(fe) => {
            let nrf = get_rollup_func(&fe.name);
            if nrf.is_none() {
                let qt_child = qt.new_child("transform %s()", fe.Name);
                let rv = eval_transform_func(qt_child, ec, fe);
                qt_child.Donef("series={}", rv.len());
                Ok(rv)
            }
            let (args, re) = eval_rollup_func_args(qt, ec, fe)?;
            let rf = nrf.unwrap()(args)?;
            return match eval_rollup_func(qt, ec, fe.Name, rf, e, &re, None) {
                Ok(res) => Ok(res),
                Err(err) => {
                    let msg = format!("cannot evaluate {}: {}", fe, err);
                    Err(Error::new(msg))
                }
            }
        }
        Expression::Aggregation(ae) => {
            let qt_child = qt.new_child("aggregate %s()", &ae.name);
            let rv = eval_aggr_func(qt_child, ec, ae);
            qt_child.Donef("series={}", len(rv));
            Ok(rv)
        }
        Expression::BinaryOperator(be) => {
            let qt_child = qt.new_child("binary op %q", be.Op);
            let rv = eval_binary_op(qt_child, ec, be);
            qt_child.Donef("series={}", len(rv));
            Ok(rv)
        }
        Expression::Number(ne) => {
            Ok(eval_number(ec, ne.N))
        }
        Expression::String(se) => {
            Ok(eval_string(ec, &se.s))
        }
        Expression::Duration(de) => {
            let d = de.duration(ec.step);
            let d_sec = d / 1000;
            Ok(eval_number(ec, d_sec))
        }
        _ => {
            fmt.Errorf("unexpected expression %q", e.AppendString(nil))
        }
    }
}

pub fn eval_transform_func(qt: &Querytracer, ec: &EvalConfig, fe: &FuncExpr) -> Result<Vec<Timeseries>> {
    let args = eval_exprs(qt, ec, &fe.args)?;
    let tf = get_transform_func(&fe.name);
    if tf.is_none() {
        let msg = format!("unknown func {}", fe.name);
        return Err(Error::from(msg));
    }
    let tfa = TransformFuncArg {
        ec,
        fe,
        args,
    };
    let rv = tf(tfa);
    if rv.is_error() {
        return fmt.Errorf(`cannot evaluate % q: %w`, fe, err);
    }
    return rv;
}

pub fn eval_aggr_func(qt: &Querytracer, ec: &EvalConfig, ae: &AggrFuncExpr) -> Result<Vec<Timeseries>> {
    let callbacks = getIncrementalAggrFuncCallbacks(ae.Name);
    if callbacks.is_some() {
        let (fe, nrf) = try_get_arg_rollup_func_with_metric_expr(ae);
        if fe.is_some() && nrf.is_some() {
            let func = nrf.unwrap();
            // There is an optimized path for calculating metricsql.AggrFuncExpr over: RollupFunc over metricsql.MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = eval_rollup_func_args(qt, ec, fe.unwrap())?;
            let rf = func(args);
            let iafc = newIncrementalAggrFuncContext(ae, callbacks);
            return eval_rollup_func(qt, ec, fe.name, rf, &ae, re, iafc);
        }
    }
    let args = eval_exprs(qt, ec, ae.args)?;
    let af = get_aggr_func(ae.name).unwrap();
    if af.is_none() {
        let err = format!("unknown func {}", ae.name);
        return Err(Error::from(err));
    }
    let afa = AggrFuncArg { ae, args, ec };
    match af(afa) {
        Ok(res) => Ok(res),
        Err(e) => {
            let res = format!("cannot evaluate {}: {}", afa, e);
            Err(Error::from(res))
        }
    }
}

fn eval_binary_op(qt: &Querytracer, ec: &EvalConfig, be: &BinaryOpExpr) -> Result<Vec<Timeseries>> {
    let bf = get_binary_op_func(be.op)?;

    let tss_left: Vec<Timeseries> = vec![];
    let tss_right: Vec<Timeseries> = vec![];

    if be.op == BinaryOp::And || be.op == BinaryOp::If {
        // Fetch right-side series at first, since it usually contains
        // lower number of time series for `and` and `if` operator.
        // This should produce more specific label filters for the left side of the query.
        // This, in turn, should reduce the time to select series for the left side of the query.
        (tss_right, tss_left) = eval_binary_op_args(qt, ec, &be.right, &be.left, be)?
    } else {
        (tss_left, tss_right) = eval_binary_op_args(qt, ec, &be.left, &be.right, be)?
    }

    let bfa = BinaryOpFuncArg {
        be,
        left,
        right,
    };

    match bf(bfa) {
        Err(err) => Err(Error::new(format!("cannot evaluate {}: {}", be, err))),
        OK(v) => v
    }
}

fn eval_binary_op_args(qt: &Querytracer,
                       ec: &EvalConfig,
                       expr_first: &Expression,
                       expr_second: &Expression,
                       be: &BinaryOpExpr) -> Result<(Vec<Timeseries>, Vec<Timeseries>)> {
    // Execute binary operation in the following way:
    //
    // 1) execute the expr_first
    // 2) get common label filters for series returned at step 1
    // 3) push down the found common label filters to expr_second. This filters out unneeded series
    //    during expr_second execution instead of spending compute resources on extracting and processing these series
    //    before they are dropped later when matching time series according to https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
    // 4) execute the expr_second with possible additional filters found at step 3
    //
    // Typical use cases:
    // - Kubernetes-related: show pod creation time with the node name:
    //
    //     kube_pod_created{namespace="prod"} * on (uid) group_left(node) kube_pod_info
    //
    //   Without the optimization `kube_pod_info` would select and spend compute resources
    //   for more time series than needed. The selected time series would be dropped later
    //   when matching time series on the right and left sides of binary operand.
    //
    // - Generic alerting queries, which rely on `info` metrics.
    //   See https://grafana.com/blog/2021/08/04/how-to-use-promql-joins-for-more-effective-queries-of-prometheus-metrics-at-scale/
    //
    // - Queries, which get additional labels from `info` metrics.
    //   See https://www.robustperception.io/exposing-the-software-version-to-prometheus
    let tss_first = evalExpr(qt, ec, expr_first);
    let mut second = &expr_second;

    if be.op != BinayOp::Or {
        // Do not pushdown common label filters from tss_first for `or` operation, since this can filter out the needed time series from tss_second.
        // See https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators for details.
        let mut lfs = get_common_label_filters(tss_first);
        lfs = metricsql.TrimFiltersByGroupModifier(lfs, be);
        second = metricsql.PushdownBinaryOpFilters(expr_second, lfs)
    }
    let tss_second = evalExpr(qt, ec, second)?;
    return Ok((tss_first, tss_second));
}

fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    let mut m: BTreeMap<String, BTreeSet<String>> = BTreeMap::with_capacity(tss.len());
    for ts in tss.iter() {
        for (key, value) in ts.metric_name.iter().enumerate() {
            if let Some(set) = m.get_mut(key) {
                set.insert(value)
            } else {
                let s = BTreeSet::from([value]);
                m.insert(key, s);
            }
        }
    }

    let mut lfs: Vec<LabelFilter> = Vec::with_capacity(m.len());
    for (key, values) in m {
        if values.len() != tss.len() {
            // Skip the tag, since it doesn't belong to all the time series.
            continue;
        }
        if values.len() > 1000 {
            // Skip the filter on the given tag, since it needs to enumerate too many unique values.
            // This may slow down the search for matching time series.
            continue;
        }

        let vals = values.iter().collect().sort();
        let mut str_value: String;
        let mut is_regex = false;

        if values.len() == 1 {
            str_value = values[0]
        } else {
            str_value = join_regexp_values(vals);
            is_regex = true;
        }
        let lf = if is_regex {
            LabelFilter::equal(key, str_value)?;
        } else {
            LabelFilter::regex_equal(key, str_value)?;
        };

        lfs.push(lf);
    }
    lfs.sort();
    return lfs;
}

#[inline]
fn get_unique_values(x: &[String]) -> Vec<String> {
    x.iter().unique().collect()
}

fn join_regexp_values(a: &Vec<String>) -> String {
    let init_size = a.iter().fold(
        0,
        |res, x| res + x.len(),
    );
    let mut res = String::with_capacity(init_size);
    for (i, s) in a.iter().enumerate() {
        let s_quoted = Regex::quote(s);
        res.push_str(s_quoted);
        if i < a.len() - 1 {
            b.push('|')
        }
    }
    return res;
}


fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> (Option<FuncExpr>, Option<NewRollupFunc>) {
    if ae.args.len() != 1 {
        return (None, None)
    }
    let e = &ae.args[0];
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    match e {
        Expression::MetricExpression(me) => {
            if me.is_empty() {
                return (None, None);
            }
            let fe = FuncExpr::new("default_rollup",vec![me]);
            let nrf = getRollupFunc(fe.name);
            return (Some(fe), Some(nrf));
        }
        Expression::Rollup(re) => {
            let is_me = is_metric_expr(re.expr);
            if !is_me || me.is_empty() || re.for_subquery {
                return (None, None);
            }
            // e = metricExpr[d]
            let fe = FuncExpr::new("default_rollup",vec![re]);
            let nrf = getRollupFunc(fe.Name);
            return (Some(fe), Some(nrf));
        }
        Expression::Function(fe) => {
            let nrf = getRollupFunc(fe.Name);
            if nrf.is_none() {
                return (None, None)
            }
            let rollup_arg_idx = metricsql.GetRollupArgIdx(fe);
            if rollup_arg_idx >= fe.args.len() {
                // Incorrect number of args for rollup func.
                return (None, None)
            }
            let arg = &fe.args[rollup_arg_idx];
            match arg {
                Expression::MetricExpression(me) => {
                    // e = rollupFunc(metricExpr)
                    let args = vec![me];
                    let f = FuncExpr::new(&fe.name, args);
                    return (Some(f), Some(nrf));
                },
                Expression::Rollup(re) => {
                    match re.expr {
                        Expression::MetricExpression(me) => {
                            if me.is_empty() || re.for_subquery() {
                                return (None, None)
                            }
                            // e = RollupFunc(metricExpr[d])
                            return (Some(fe), Some(nrf));
                        },
                        _ => (None, None)
                    }
                },
                _ =>  return (None, None)
            }
        },
        _ => return (None, None)
    }
}

fn eval_exprs(qt: &QueryTracer, ec: &EvalConfig, es: &Vec<Expression>) -> Result<Vec<Vec<Timeseries>>> {
    let mut rvs: Vec<Vec<Timeseries>> = Vec::with_capacity(es.len());
    for e in es {
        let rv = evalExpr(qt, ec, e)?;
        rvs.push(rv);
    }
    Ok(rvs)
}

fn is_metric_expression(e: &Expression) -> Option<&MetricExpr> {
    match e {
        Expression::MetricExpression(me) => Some(me),
        _ => None
    }
}

///////////////////////////////////////

fn eval_rollup_func_args(qt: &QueryTracer, ec: &EvalConfig, fe: &FuncExpr) -> Result<(Vec<interface>, RollupExpr)> {
    let re: RollupExpr;
    let rollup_arg_idx = metricsql.GetRollupArgIdx(fe);
    if re.args.len() <= rollup_arg_idx {
        let err = format!("expecting at least {} args to {}; got {} args; expr: {}",
                          rollup_arg_idx + 1, fe.name, re.args.len(), fe);
        return Err(Error::from(err));
    }
    let mut args = Vec::with_capacity(re.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg);
            args[i] = re;
            continue;
        }
        let ts = eval_expr(qt, ec, arg);
        if ts.is_error() {
            let msg = format!("cannot evaluate arg #{} for {}: {}", i + 1, fe, err);
            return Err(Error::new(msg));
        }
        args[i] = ts.unwrap();
    }
    return Ok((args, re));
}

// todo: move to optimize phase
fn get_rollup_expr_arg(arg: Expression) -> RollupExpr {
    let mut re: RollupExpr = match arg { 
        Expression::Rollup(re) => re,
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            RollupExpr::new(arg)
        }
    };
    if !re.for_subquery() {
        // Return standard rollup if it doesn't contain subquery.
        return re;
    }
    return match re.expr {
        Expression::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]
            let re_new = *re;
            let rollup = RollupExpr::new(me);
            re_new.expr = FuncExpr::new("default_rollup", vec![rollup]);
            &re_new
        },
        _ => {
            // arg contains subquery.
            re
        }
    }
}

// expr may contain:
// -: RollupFunc(m) if iafc is nil
// - aggrFunc(rollupFunc(m)) if iafc isn't nil
fn eval_rollup_func(qt: &QueryTracer,
                    ec: &EvalConfig,
                    func_name: &str,
                    rf: RollupFunc,
                    expr: &Expression,
                    re: &RollupExpr,
                    iafc: &IncrementalAggrFuncContext) -> Result<Vec<Timeseries>> {
    if re.at.is_none() {
        return eval_rollup_func_without_at(qt, ec, func_name, rf, &expr, &re, iafc);
    }
    let tss_at = match evalExpr(qt, ec, re.At) {
        Err(err) => {
            let e = format!("cannot evaluate `@` modifier: {}", err);
            Err(Error::new(msg))
        },
        Ok(res) => res
    };

    if tss_at.is_error() {
        return tss_at;
    }

    let tss_at = tss_at.unwrap();

    if tss_at.len() != 1 {
        let msg = format!("`@` modifier must return a single series; it returns {} series instead", tss_at.len());
        return Err(Error::new(msg));
    }
    let at_timestamp = tss_at[0].values[0] * 1000 as i64;
    let ec_new = copyEvalConfig(ec);
    ec_new.start = at_timestamp;
    ec_new.end = at_timestamp;
    let tss = eval_rollup_func_without_at(qt, ec_new, func_name, rf, expr, re, iafc)?;

    // expand single-point tss to the original time range.
    let timestamps = ec.getSharedTimestamps();
    for mut ts in tss {
        let v = ts.values[0];
        let mut values = Vec::with_capacity(timestamps.len());
        for i in 0..timestamps.len() {
            values[i] = v
        }
        ts.timestamps = RC::clone(&timestamps);
        ts.values = values;
    }
    return Ok(tss);
}

fn eval_rollup_func_without_at(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: &Expression,
    re: &RollupExpr,
    iafc: IncrementalAggrFuncContext) -> Result<Vec<Timeseries>> {

    let f_name = func_name.to_lowercase().as_str();

    let mut ec_new = ec;
    let mut offset: i64 = 0;
    if re.offset.is_some() {
        offset = re.offset.duration(ec.step);
        ec_new = copyEvalConfig(ec_new);
        ec_new.start = ec_new.start - offset;
        ec_new.end = ec_new.end - offset;
        // There is no need in calling AdjustStartEnd() on ec_new if ec_new.MayCache is set to true,
        // since the time range alignment has been already performed by the caller,
        // so cache hit rate should be quite good.
        // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
    }
    if func_name == "rollup_candlestick" {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        let step = ec_new.step;
        ec_new = copyEvalConfig(ec_new);
        ec_new.start = ec_new.start + step;
        ec_new.end = ec_new.end + step;
        offset = offset - step;
    }
    let mut rvs: Vec<Timeseries>;

    let mut rvs = match re.expr {
        Expression::MetricExpression(me) => {
            eval_rollup_func_with_metric_expr(qt, ec_new, func_name, rf, expr, me, iafc, re.window)?;
        }
        _ => {
            if Some(iafc) {
                let msg = format!("BUG: iafc must be nil for rollup {} over subquery {}", func_name, re);
                return Err(Error::new(msg));
            }
            eval_rollup_func_with_subquery(qt, ec_new, func_name, rf, expr, re)?;
        }
    };

    if func_name == "absent_over_time" {
        rvs = aggregate_absent_over_time(ec, re.expr, &rvs)
    }
    if offset != 0 && rvs.len() > 0 {
        // Make a copy of timestamps, since they may be used in other values.
        let src_timestamps = rvs[0].timestamps;
        let mut dst_timestamps = Vec::with_capacity(src_timestamps.len());
        for i in 0..dst_timestamps.len() {
            dst_timestamps[i] += offset;
        }
        for ts in rvs {
            ts.timestamps = dst_timestamps
        }
    }
    return rvs;
}

// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
//
// Values for returned series are set to nan if at least a single tss series contains nan at that point.
// This means that tss contains a series with non-empty results at that point.
// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(ec: &EvalConfig, expr: &Expression, tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs = getAbsentTimeseries(ec, expr);
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

fn eval_rollup_func_with_subquery(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: &Expression,
    re: &RollupExpr) -> Result<Vec<Timeseries>> {
// TODO: determine whether to use rollupResultCacheV here.
    qt = qt.new_child("subquery")
    defer
    qt.Done()

    let mut step = re.step.duration(ec.step);
    if step == 0 {
        step = ec.step
    }
    let window = re.window.duration(ec.step);

    let mut ecSQ = copyEvalConfig(ec);
    ecSQ.start -= window + maxSilenceInterval + step;
    ecSQ.end += step;
    ecSQ.step = step;
    validateMaxPointsPerTimeseries(ecSQ.start, ecSQ.end, ecSQ.step)?;

    // unconditionally align start and end args to step for subquery as Prometheus does.
    (ecSQ.start, ecSQ.end) = align_start_end(ecSQ.start, ecSQ.end, ecSQ.step);
    let tss_sq = eval_expr(qt, ecSQ, re.expr)?;
    if tss_sq.len() == 0 {
        return None;
    }
    let shared_timestamps = get_timestamps(&ec.start, &ec.end, &ec.step);
    let (pre_func, rcs) = getRollupConfigs(
        func_name,
        rf,
        expr,
        &ec.start,
        &ec.end,
        &ec.step,
        window,
        &ec.lookback_delta,
        shared_timestamps)?;

    let tss: Vec<Timeseries> = Vec::with_capacity(tss_sq.len() * rcs.len());
    let tss_lock: sync.Mutex;
    let keep_metric_names = get_keep_metric_names(expr);

    doParallel(tss_sq, |tsSQ: Timeseries, values: &[f64], timestamps: &[i64]| -> (Vec<f64>, Vec<i64>) {
        let (values, timestamps) = remove_nan_values(values[: 0],
                                                     &timestamps,
                                                     &tsSQ.values,
                                                     &tsSQ.timestamps)
        pre_func(values, timestamps);
        for _rc in rcs {
            let tsm = newTimeseriesMap(funcName: func_name, keep_metric_names, shared_timestamps, &tsSQ.MetricName);
            if Some(tsm) {
                rc.DoTimeseriesMap(tsm, values, timestamps);
                tss_lock.Lock();
                tss = tsm.AppendTimeseriesTo(tss);
                tss_lock.Unlock();
                continue;
            }
            let ts: Timeseries;
            do_rollup_for_timeseries(
                func_name,
                keep_metric_names,
                rc,
                &ts,
                &tsSQ.MetricName,
                values,
                timestamps,
                shared_timestamps);
            tss_lock.Lock();
            tss.append(ts);
            tss_lock.Unlock()
        }
        return (values, timestamps);
    });
    qt.Printf("rollup %s() over {} series returned by subquery: series={}", funcName, len(tss_sq), len(tss))
    return tss;, nil
}

fn get_keep_metric_names(expr: &Expression) -> bool {
    // todo: move to optimize stage. put result in ast node
    match expr {
        Expression::Aggregation(ae) => {
            *ae.keep_metric_names
        },
        Expression::Function(fe) => {
            *fe.keep_metric_names
        }
        _ => false
    }
}

fn doParallel(tss: &Vec<Timeseries>, f: fn(ts: &Timeseries, values: &[f64], &[i64]) -> (Vec<f64>, Vec<i64>)) {
    let concurrency = cgroup.AvailableCPUs();
    workCh = make(chan * timeseries, concurrency)
    let wg: sync.WaitGroup;
    wg.Add(concurrency)
    for i in 0..concurrency {
        go
        func()
        {
            defer
            wg.Done()
            let tmpValues: Vec<f64>
            let tmpTimestamps: []int64
            for ts: = range workCh {
                tmpValues,
                tmpTimestamps = f(ts, tmpValues, tmpTimestamps)
            }
        }
        ()
    }
    for ts in tss {
        workCh < -ts
    }
    close(workCh)
    wg.Wait()
}

fn remove_nan_values(dst_values: &mut Vec<f64>, mut dst_timestamps: &Vec<i64>, values: &[f64], timestamps: &[i64]) -> (Vec<f64>, Vec<i64>) {
    let mut has_nan = false;
    for v in values {
        if v.is_nan() {
            has_nan = true
        }
    }
    if !has_nan {
        // Fast path - no NaNs.
        dst_values.append(values.iter().cloned());
        dst_timestamps.append(timestamps.iter().cloned());
        return (dst_values, dst_timestamps);
    }

// Slow path - remove NaNs.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        dst_values.push(v);
        dst_timestamps.push(timestamps[i])
    }
    return (dst_values, dst_timestamps);
}

var (
rollupResultCacheFullHits    = metrics.NewCounter(`vm_rollup_result_cache_full_hits_total`)
rollupResultCachePartialHits = metrics.NewCounter(`vm_rollup_result_cache_partial_hits_total`)
rollupResultCacheMiss        = metrics.NewCounter(`vm_rollup_result_cache_miss_total`)
)

fn eval_rollup_func_with_metric_expr(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: &Expression,
    me: &MetricExpr,
    iafc: &IncrementalAggrFuncContext,
    window_expr: &DurationExpr) -> Result<Vec<Timeseries>> {
    let mut rollup_memory_size: i64;

    let window = window_expr.Duration(ec.Step);
    qt = qt.new_child("rollup %s(): timeRange=[{}..{}], step={}, window={}", func_name, ec.Start, ec.End, ec.Step, window)

    defer
    func()
    {
        qt.Donef("neededMemoryBytes={}", rollup_memory_size)
    }
    ()
    if me.IsEmpty() {
        return eval_number(ec, nan);
    }

    // Search for partial results in cache.
    let (tss_cached, start) = rollupResultCacheV.Get(qt, ec, expr, window);
    if start > ec.end {
        // The result is fully cached.
        rollupResultCacheFullHits.Inc()
        return tss_cached;
    }
    if start > ec.start {
        rollupResultCachePartialHits.Inc()
    } else {
        rollupResultCacheMiss.Inc()
    }

    // Obtain rollup configs before fetching data from db,
    // so type errors can be caught earlier.
    let shared_timestamps = getTimestamps(start, ec.end, ec.step);
    let (preFunc, rcs) = getRollupConfigs(func_name, rf, expr, start, ec.end, ec.step, window, ec.lookbackDelta, shared_timestamps)?;

    // Fetch the remaining part of the result.
    let tfs = searchutils.ToTagFilters(me.LabelFilters);
    let tfss = searchutils.JoinTagFilterss([][]storage.TagFilter{ tfs }, ec.EnforcedTagFilterss)
    let min_timestamp = start - maxSilenceInterval;
    if window > ec.Step {
        min_timestamp -= window
    } else {
        min_timestamp -= ec.Step
    }
    let sq = storage.NewSearchQuery(min_timestamp, ec.end, tfss, ec.maxSeries);
    let rss = netstorage.ProcessSearchQuery(qt, sq, true, ec.Deadline)?;
    let rss_len = rss.Len();
    if rss_len == 0 {
        rss.cancel();
        tss = mergeTimeseries(tss_cached, None, start, ec);
        return tss;
    }

    // Verify timeseries fit available memory after the rollup.
    // Take into account points from tss_cached.
    let points_per_timeseries = 1 + (ec.end - ec.start) / ec.step;
    let mut timeseries_len = rss_len;
    if iafc != nil {
        // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
        timeseries_len = cgroup.AvailableCPUs();
        if iafc.ae.modifier.Op != "" {
            if iafc.ae.Limit > 0 {
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
    let rollup_points = mulNoOverflow(points_per_timeseries, int64(timeseries_len * len(rcs)));
    let rollupMemorySize = mulNoOverflow(rollup_points, 16);
    let rml = getRollupMemoryLimiter();
    if !rml.Get(uint64(rollupMemorySize)) {
        rss.Cancel();
        let msg = format!("not enough memory for processing {} data points across {} time series with {} points in each time series; " +
                                    "total available memory for concurrent requests: {} bytes; " +
                                    "requested memory: {} bytes; " +
                                    "possible solutions are: reducing the number of matching time series; switching to node with more RAM; " +
                                    "increasing -memory.allowedPercent; increasing `step` query arg ({})",
                          rollup_points, timeseries_len * len(rcs), points_per_timeseries, rml.MaxSize, uint64(rollupMemorySize), float64(ec.Step) / 1e3)
    }
    defer
    rml.Put(uint64(rollupMemorySize));

// Evaluate rollup
    let keep_metric_names = get_keep_metric_names(expr);
    let mut tss: &Vec<Timeseries>;
    if iafc != nil {
        tss = evalRollupWithIncrementalAggregate(qt, &func_name, keep_metric_names, iafc, rss, rcs, preFunc, shared_timestamps)?;
    } else {
        tss = eval_rollup_no_incremental_aggregate(qt, &func_name, keep_metric_names, rss, rcs, preFunc, shared_timestamps)?;
    }

    tss = mergeTimeseries(tss_cached, tss, start, ec);
    rollupResultCacheV.Put(qt, ec, expr, window, tss);
    return tss;
}

var (
rollupMemoryLimiter     memoryLimiter
rollupMemoryLimiterOnce sync.Once
)

fn getRollupMemoryLimiter() * memoryLimiter {
rollupMemoryLimiterOnce.Do(func() {
rollupMemoryLimiter.MaxSize = uint64(memory.Allowed()) / 4
})
return & rollupMemoryLimiter
}

fn evalRollupWithIncrementalAggregate(qt: &QueryTracer,
                                      func_name: String,
                                      keep_metric_names: bool,
                                      iafc: IncrementalAggrFuncContext,
                                      rss: Results,
                                      rcs: &[RollupConfig],
                                      pre_func: fn(values: &[f64], timestamps: &[i64]),
                                      shared_timestamps: &[i64]) -> Result<Vec<Timeseries>> {
    let mut qt_ = qt.new_child("rollup %s() with incremental aggregation %s() over {} series", func_name, iafc.ae.Name, rss.Len())
    // defer qt.Done()

    RunParallel(qt, |rs: QueryResult, workerID: usize| {
        (rs.values, rs.timestamps) = drop_stale_nans(func_name, rs.values, rs.timestamps);
        pre_func(rs.calues, rs.timestamps);
        let ts = getTimeseries()
        defer
        putTimeseries(ts)
        for rc in rcs {
            let tsm = newTimeseriesMap(funcName, keepMetricNames, shared_timestamps, &rs.metric_name);
            if Some(tsm) {
                rc.DoTimeseriesMap(tsm, rs.values, rs.timestamps);
                for ts in tsm.m {
                    iafc.updateTimeseries(ts, workerID)
                }
                continue;
            }
            ts.reset();
            do_rollup_for_timeseries(funcName, keepMetricNames, rc, ts, &rs.MetricName, rs.Values, rs.Timestamps, shared_timestamps)
            iafc.updateTimeseries(ts, workerID)

            // ts.Timestamps points to shared_timestamps. Zero it, so it can be re-used.
            ts.timestamps = nil
            ts.denyReuse = false
        }
        return nil;
    };

    let tss = iafc.finalizeTimeseries();
    qt.Printf("series after aggregation with %s(): {}", iafc.ae.Name, len(tss))
    return tss;
}

fn eval_rollup_no_incremental_aggregate(
    qt: &QueryTracer,
    func_name: String,
    keep_metric_names: bool,
    rss: Results,
    rcs: Vec<RollupConfig>,
    pre_func: fn(values: &[f64], timestamps: &[i64], shared_timestamps: &[i64]),
) -> Result<Vec<Timeseries>> {
    let qt = qt.new_child("rollup %s() over {} series", func_name, rss.len());

    defer
    qt.Done()

    let tss: Vec<Timeseries> = Vec::with_capacity(rss.len() * rcs.len());

    let tssLock: Arc<Mutex<Vec<Timeseries>>> = Arc::new(Mutex::new(tss));

    rss.RunParallel(qt, move |rs: &QueryResult, workerID: uint | {
        (rs.values, rs.timestamps) = drop_stale_nans(func_name, rs.values, rs.timestamps);
        pre_func(rs.values, rs.timestamps);
        for rc in rcs {
            let tsm = newTimeseriesMap(func_name, keep_metric_names, sharedTimestamps, &rs.metric_name);
            if Some(tsm) {
                rc.doTimeseriesMap(tsm, rs.values, rs.timestamps);
                let _tss = tssLock.unlock().unwrap();
                tsm.append_timeseries_to(_tss);
                continue;
            }
            let ts: Timeseries;
            do_rollup_for_timeseries(
                func_name,
                keep_metric_names,
                rc,
                &ts,
                &rs.metric_name,
                rs.values,
                rs.timestamps,
                sharedTimestamps);
            let _tss = tssLock.unlock().unwrap();
            _tss.push(ts);
        }
    });

    return tss;
}

fn do_rollup_for_timeseries(func_name: &str,
                            keep_metric_names: bool,
                            rc: RollupConfig,
                            ts_dst: &Timeseries,
                            mn_src: &MetricName,
                            values_src: Vec<f64>,
                            timestamps_src: Vec<i64>,
                            shared_timestamps: Vec<i64>) {
    ts_dst.metric_name.copy_from(mn_src);
    if rc.tag_value.len() > 0 {
        ts_dst.metric_name.add_tag("rollup", rc.TagValue)
    }
    if !keep_metric_names && !rollupFuncsKeepMetricName[func_name] {
        ts_dst.MetricName.ResetMetricGroup()
    }
    ts_dst.values = rc.Do(ts_dst.values, values_src, timestamps_src);
    ts_dst.timestamps = shared_timestamps;
    ts_dst.denyReuse = true
}

var bbPool bytesutil.ByteBufferPool

fn eval_number(ec: &EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.getSharedTimestamps();
    let mut values = Vec::with_capacity(timestamps.len());
    for i in 0..timestamps.len() {
        values[i] = n;
    }
    let ts = Timeseries::with_shared_timestamps(timestamps, values);
    return vec![ts];
}

fn eval_string(ec: &EvalConfig, s: String) -> Vec<Timeseries> {
    let mut rv = eval_number(ec, nan);
    rv[0].metric_name.metric_group = s;
    return rv;
}

fn eval_time(ec: &EvalConfig) -> Vec<Timeseries> {
    let rv = eval_number(ec, nan);
    let timestamps = rv[0].timestamps;
    let mut values = &rv[0].values;
    for (i, ts) in timestamps.iter().enumerate() {
        values[i] = (ts / 1e3) as f64;
    }
    return rv;
}

fn mulNoOverflow(a: i64, b: i64) -> i64 {
    if math.MaxInt64 / b < a {
        // Overflow
        return math.MaxInt64;
    }
    return a * b;
}

fn drop_stale_nans(func_name: String, mut values: Vec<f64>, mut timestamps: &mut Vec<i64>) -> (Vec<f64>, Vec<i64>) {
    if *noStaleMarkers || func_name == "default_rollup" || func_name == "stale_samples_over_time" {
        // Do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // Do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return (values, timestamps);
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = values.iter().any(|x| lib::decimal::is_stale_nan(x));

    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return (values, timestamps);
    }
    // Slow path: drop Prometheus staleness marks from values.
    let mut dst_values = values;

    let mut k = 0;
    for (i, v) in values.iter_into().enumerate() {
        if is_stale_nan(v) {
            continue;
        }
        dst_values[k] = v;
        timestamps[k] = timestamps[i];
        k = k + 1;
    }
    return (dst_values, dst_timestamps);
}