use std::collections::{BTreeMap, BTreeSet};
use regex::Regex;
use lib::error::Error;
use metricsql::types::{AggrFuncExpr, BinaryOp, BinaryOpExpr, DurationExpr, Expression, FuncExpr, MetricExpr, RollupExpr};
use super::binary_op::*;

// The minimum number of points per timeseries for enabling time rounding.
// This improves cache hit ratio for frequently requested queries over
// big time ranges.
const MinTimeseriesPointsForTimeRounding: usize = 50;

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
    if points < MinTimeseriesPointsForTimeRounding {
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

fn align_start_end(start: i64, end: i64, step: i64) -> (i64, i64) {
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

    pub fn validate(&self) -> Result<(), Error> {
        if self.start > self.end {
            logger.Panicf("BUG: start cannot exceed end; got %d vs %d", ec.Start, ec.End)
        }
        if self.step <= 0 {
            logger.Panicf("BUG: step must be greater than 0; got %d", ec.Step)
        }
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

fn get_timestamps(start: i64, end: i64, step: i64) -> Result<Vec<i64>, Error> {
    // Sanity checks.
    if step <= 0 {
        logger.Panicf("BUG: Step must be bigger than 0; got %d", step)
    }
    if start > end {
        logger.Panicf("BUG: Start cannot exceed End; got %d vs %d", start, end)
    }
    match validateMaxPointsPerTimeseries(start, end, step) {
        Err(err) => {
            logger.Panicf("BUG: %s; this must be validated before the call to get_timestamps", err)
        }
        _ => ()
    }

    // Prepare timestamps.
    let n = 1 + (end - start) / step;
    let mut timestamps: Vec<i64> = Vec::create_with_capacity(n);
    let mut cursor = start;
    for i in 0..n {
        timestamps.push(start);
        cursor = cursor + step
    }
    return Ok(timestamps);
}

pub fn eval_expr(qt: &QueryTracer, ec: &EvalConfig, e: &Expression) -> Result<Vec<Timeseries>, Error> {
    if qt.enabled() {
        let may_cache = ec.mayCache();
        qt = qt.new_child("eval: query={}, timeRange=[{}..{}}], step={}, may_cache={}", e, ec.start, ec.end, ec.step, may_cache)
    }
    let rv = eval_expr_internal(qt, ec, e);
    if qt.enabled() {
        seriesCount = rv.len();
        let mut points_per_series = 0;
        if rv.len() > 0 {
            points_per_series = rv[0].timestamps.len();
        }
        let points_count = seriesCount * points_per_series;
        qt.Donef("series={}, points={}, points_per_series={}", seriesCount, points_count, points_per_series)
    }
    return rv;
}

pub fn eval_expr_internal(qt: &Querytracer, ec: &EvalConfig, e: &MetricExpr) -> Result<Vec<Timeseries>, Error> {
    match e {
        Expression::MetricExpression(me) => {
            let re = RollupExpr::new(me);
            match evalRollupFunc(qt, ec, "default_rollup", rollupDefault, e, re, None) {
                Err(err) => {
                    fmt.Errorf("cannot evaluate {}: {}", me, err)
                }
                Ok(res) => res
            }
        }
        Expression::Rollup(re) => {
            match evalRollupFunc(qt, ec, "default_rollup", rollupDefault, e, re, None) {
                Err(err) => {
                    fmt.Errorf("cannot evaluate {}: {}", me, err)
                }
                Ok(res) => res
            }
        }
        Expression::FuncExpr(fe) => {
            let nrf = getRollupFunc(fe.name);
            if nrf.is_none() {
                let qt_child = qt.new_child("transform %s()", fe.Name);
                let rv = evalTransformFunc(qt_child, ec, fe);
                qt_child.Donef("series=%d", rv.len());
                return rv;
            }
            let (args, re) = evalRollupFuncArgs(qt, ec, fe)?;
            let rf = nrf(args)?;
            let rv = evalRollupFunc(qt, ec, fe.Name, rf, e, re, None);
            if err != nil {
                return nil;, fmt.Errorf(`cannot evaluate % q: %w`, fe.AppendString(nil), err)
            }
            return rv;
        }
        Expression::Aggregation(ae) => {
            let qt_child = qt.NewChild("aggregate %s()", ae.name);
            rv = evalAggrFunc(qt_child, ec, ae);
            qt_child.Donef("series=%d", len(rv));
            return rv;
        }
        Expression::BinaryOperator(be) => {
            let qt_child = qt.NewChild("binary op %q", be.Op);
            let rv = eval_binary_op(qt_child, ec, be);
            qt_child.Donef("series=%d", len(rv));
            return Ok(rv);
        }
        Expression::Number(ne) => {
            Ok(evalNumber(ec, ne.N))
        }
        Expression::StringExpr(se) => {
            Ok(evalString(ec, se.s))
        }
        Expression::DurationExpr(de) => {
            let d = de.Duration(ec.Step);
            let d_sec = float64(d) / 1000;
            return evalNumber(ec, d_sec);
        }
        _ => {
            fmt.Errorf("unexpected expression %q", e.AppendString(nil))
        }
    }
}

pub fn eval_transform_func(qt: &Querytracer, ec: &EvalConfig, fe: &FuncExpr) -> Result<Vec<Timeseries>, Error> {
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

pub fn eval_aggr_func(qt: &Querytracer, ec: &EvalConfig, ae: &AggrFuncExpr) -> Result<Vec<Timeseries>, Error> {
    let callbacks = getIncrementalAggrFuncCallbacks(ae.Name);
    if callbacks.is_some() {
        let (fe, nrf) = try_get_arg_rollup_func_with_metric_expr(ae);
        if fe.is_some() && nrf.is_some() {
            let func = nrf.unwrap();
            // There is an optimized path for calculating metricsql.AggrFuncExpr over: RollupFunc over metricsql.MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = evalRollupFuncArgs(qt, ec, fe.unwrap())?;
            let rf = func(args);
            let iafc = newIncrementalAggrFuncContext(ae, callbacks);
            return evalRollupFunc(qt, ec, fe.Name, rf, ae, re, iafc);
        }
    }
    let args = eval_exprs(qt, ec, ae.args)?;
    af = get_aggr_func(ae.name).unwrap();
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

fn eval_binary_op(qt: &Querytracer, ec: &EvalConfig, be: &BinaryOpExpr) -> Result<Vec<Timeseries>, Error> {
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
                       be: &BinaryOpExpr) -> Result<(Vec<Timeseries>, Vec<Timeseries>), Error> {
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
            LabelFilter::equal(key, str_value);
        } else {
            LabelFilter::regex_equal(key, str_value);
        };

        lfs.push(lf);
    }
    lfs.sort();
    return lfs;
}


fn get_unique_values(a: &Vec<String>) -> Vec<String> {
    let dedup: Vec<_> = x.iter().unique().collect();
    return dedup;
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
    let e = ae.args[0];
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
            let fe = Metricsql::FuncExpr {
                name: "default_rollup",
                args: Vec![me],
            };
            let nrf = getRollupFunc(fe.name);
            return (Some(fe), Some(nrf));
        }
        Expression::Rollup(re) => {
            let is_me = is_metric_expr(re.expr);
            if !is_me || me.is_empty() || re.for_subquery {
                return (None, None);
            }
            // e = metricExpr[d]
            let fe = FuncExpr {
                name: "default_rollup",
                args: Vec![re],
                keep_metric_names: false
            };
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
            let arg = fe.args[rollup_arg_idx];
            match arg {
                Expression::MetricExpression(me) => {
                    // e = rollupFunc(metricExpr)
                    let args = vec![me];
                    let f = FuncExpr::new(fe.name, args);
                    return (Some(f), Some(nrf));
                },
                Expression::Rollup(re) => {
                    match *re.expr {
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

fn eval_exprs(qt: &QueryTracer, ec: &EvalConfig, es: &Vec<Expression>) -> Result<Vec<Vec<Timeseries>>, Error> {
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

fn evalRollupFuncArgs(qt: &QueryTracer, ec: &EvalConfig, fe: &FuncExpr) -> Result<(Vec<interface>, RollupExpr), Error> {
    let re: RollupExpr;
    let rollupArgIdx = metricsql.GetRollupArgIdx(fe);
    if re.args.len() <= rollupArgIdx {
        let err = format!("expecting at least {} args to {}; got {} args; expr: {}", 
                          rollupArgIdx + 1, fe.name, re.args.len(), fe);
        return Err(Error::from(err));
    }
    let mut args = Vec::with_capacity(re.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollupArgIdx {
            re = getRollupExprArg(arg);
            args[i] = re;
            continue;
        }
        let ts = evalExpr(qt, ec, arg);
        if err != nil {
            return nil;, nil, fmt.Errorf("cannot evaluate arg #%d for %q: %w", i + 1, fe.AppendString(nil), err)
        }
        args[i] = ts
    }
    return Ok((args, re));
}

fn getRollupExprArg(arg: Expression) -> RollupExpr {
    let mut re: RollupExpr = match arg { 
        Expression::Rollup(re) => re,
        _ => {
            // Wrap non-rollup arg into metricsql.RollupExpr.
            RollupExpr::new(arg)
        }
    };
    if !re.for_subquery() {
        // Return standard rollup if it doesn't contain subquery.
        return re;
    }
    match re.expr {
        Expression::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]
            let reNew = *re;
            reNew.expr = FuncExpr::new()
            {
                name: "default_rollup",
                args: vec![],
                expr: Box::new(RollupExpr::new(me)),
            }
            return &reNew;            
        },
        _ => {
            // arg contains subquery.
            return re;            
        }
    }
}

// expr may contain:
// -: RollupFunc(m) if iafc is nil
// - aggrFunc(rollupFunc(m)) if iafc isn't nil
fn evalRollupFunc(qt: &QueryTracer,
                  ec: &EvalConfig,
                  func_name: &str,
                  rf: RollupFunc,
                  expr: Expression,
                  re: &RollupExpr,
                  iafc: &IncrementalAggrFuncContext) -> Result<Vec<Timeseries>, Error> {
    if re.at.is_none() {
        return evalRollupFuncWithoutAt(qt, ec, func_name, rf, expr, re, iafc);
    }
    let tssAt = match evalExpr(qt, ec, re.At) {
        Err(err) = {
        let e = format ! ("cannot evaluate `@` modifier: {}", err);
        // err
        return err;
        },
        Ok(res) => res
    }

    if tssAt.is_error() {
        return tssAt;
    }

    if tssAt.len() != 1 {
        return nil;, fmt.Errorf("`@` modifier must return a single series; it returns %d series instead", len(tssAt))
    }
    let at_timestamp = tssAt[0].values[0] * 1000 as i64;
    let ecNew = copyEvalConfig(ec);
    ecNew.start = at_timestamp;
    ecNew.end = at_timestamp;
    let tss = evalRollupFuncWithoutAt(qt, ecNew, func_name, rf, expr, re, iafc)?;

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

fn evalRollupFuncWithoutAt(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: Expression,
    re: RollupExpr, iafc: IncrementalAggrFuncContext) -> Result<Vec<Timeseries>, Error> {
    func_name = Strings.ToLower(func_name)
    let ecNew = ec;
    let mut offset: i64;
    if Some(re.offset) {
        let offset = re.Offset.Duration(ec.step);
        let mut ecNew = copyEvalConfig(ecNew);
        ecNew.start -= offset;
        ecNew.end -= offset;
        // There is no need in calling AdjustStartEnd() on ecNew if ecNew.MayCache is set to true,
        // since the time range alignment has been already performed by the caller,
        // so cache hit rate should be quite good.
        // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
    }
    if func_name == "rollup_candlestick" {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        let step = ecNew.step;
        let mut ecNew = copyEvalConfig(ecNew);
        ecNew.start += step;
        ecNew.end += step;
        offset -= step;
    }
    let mut rvs: Vec<Timeseries>;

    let rvs = match re.expr {
        Expression::MetricExpr(me) => {
            evalRollupFuncWithMetricExpr(qt, ecNew, func_name, rf, expr, me, iafc, re.window)?;
        }
        _ => {
            if Some(iafc) {
                logger.Panicf("BUG: iafc must be nil for rollup %q over subquery %q", func_name, re.AppendString(nil))
            }
            evalRollupFuncWithSubquery(qt, ecNew, func_name, rf, expr, re)?;
        }
    };

    if func_name == "absent_over_time" {
        rvs = aggregateAbsentOverTime(ec, re.expr, rvs)
    }
    if offset != 0 && rvs.len() > 0 {
        // Make a copy of timestamps, since they may be used in other values.
        srcTimestamps = rvs[0].timestamps;
        dstTimestamps = append([]int64 {}, srcTimestamps...)
        for i in 0..dstTimestamps.len() {
            dstTimestamps[i] += offset,
        }
        for ts in rvs {
            ts.Timestamps = dstTimestamps
        }
    }
    return rvs;
}

// aggregateAbsentOverTime collapses tss to a single time series with 1 and nan values.
//
// Values for returned series are set to nan if at least a single tss series contains nan at that point.
// This means that tss contains a series with non-empty results at that point.
// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregateAbsentOverTime(ec: &EvalConfig, expr: Expression, tss: &Vec<Timeseries>) -> Vec<Timeseries> {
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

fn evalRollupFuncWithSubquery(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: Expression,
    re: &RollupExpr) -> Result<Vec<Timeseries>, Error> {
// TODO: determine whether to use rollupResultCacheV here.
    qt = qt.NewChild("subquery")
    defer
    qt.Done()

    let mut step = re.step.Duration(ec.step);
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
    (ecSQ.start, ecSQ.end) = alignStartEnd(ecSQ.start, ecSQ.end, ecSQ.step);
    let tssSQ = evalExpr(qt, ecSQ, re.Expr)?;
    if tssSQ.len() == 0 {
        return None;
    }
    let shared_timestamps = get_timestamps(ec.start, ec.end, ec.step);
    let (preFunc, rcs) = getRollupConfigs(func_name, rf, expr, ec.Start, ec.End, ec.Step, window, ec.LookbackDelta, shared_timestamps)?;

    let tss: Vec<Timeseries> = Vec::with_capacity(tssSQ.len() * rcs.len());
    let tssLock: sync.Mutex;
    let keep_metric_names = getKeepMetricNames(expr);

    doParallel(tssSQ, |tsSQ: Timeseries, values: Vec<f64>, timestamps: Vec<i64>| -> (Vec<f64>, Vec<i64>) {
        let (values, timestamps) = removeNanValues(values[: 0], timestamps[: 0], tsSQ.Values, tsSQ.Timestamps)
        preFunc(values, timestamps);
        for _rc in rcs {
            let tsm = newTimeseriesMap(funcName: func_name, keep_metric_names, shared_timestamps, &tsSQ.MetricName);
            if Some(tsm) {
                rc.DoTimeseriesMap(tsm, values, timestamps);
                tssLock.Lock();
                tss = tsm.AppendTimeseriesTo(tss)
                tssLock.Unlock();
                continue;
            }
            let ts: Timeseries;
            doRollupForTimeseries(
                func_name,
                keep_metric_names,
                rc,
                &ts,
                &tsSQ.MetricName,
                values,
                timestamps,
                shared_timestamps);
            tssLock.Lock();
            tss.append(ts);
            tssLock.Unlock()
        }
        return (values, timestamps);
    });
    qt.Printf("rollup %s() over %d series returned by subquery: series=%d", funcName, len(tssSQ), len(tss))
    return tss;, nil
}

fn getKeepMetricNames(expr: &Expression) -> bool {
    // todo: move to optimize stage. put result in ast node
    let mut mExpr = expr;
    match expr {
        Expression::AggrFuncExpr(ae) => {
            // Extract: RollupFunc(...) from aggrFunc(rollupFunc(...)).
            // This case is possible when optimized aggrfn calculations are used
            // such as `sum(rate(...))`
            if ae.args.len() != 1 {
                return false;
            }
            mExpr = ae.args[0];
        }
        _ => ()
    }
    match mExpr {
        Expression::Function(fe) => {
            fe.keep_metric_namee
        }
        _ => false
    }
}

fn doParallel(tss: &Vec<Timeseries>, f: fn(ts: &Timeseries, values: Vec<f64>, timestamps: Vec<i64>) -> (Vec<f64>, Vec<i64>)) {
    let concurrency = cgroup.AvailableCPUs();
    if concurrency > len(tss) {
        concurrency = len(tss)
    }
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

fn removeNanValues(dstValues: Vec<f64>, mut dstTimestamps: Vec<i64>, values: &[f64], timestamps: &[i64]) -> (Vec<f64>, Vec<i64>) {
    let mut hasNan = false;
    for v in values {
        if v.is_nan() {
            hasNan = true
        }
    }
    if !hasNan {
        // Fast path - no NaNs.
        dstValues = append(dstValues, values...)
        dstTimestamps = append(dstTimestamps, timestamps...)
        return dstValues;, dstTimestamps
    }

// Slow path - remove NaNs.
    for (i, v) in values {
        if math.IsNaN(v) {
            continue;
        }
        dstValues = append(dstValues, v)
        dstTimestamps = append(dstTimestamps, timestamps[i])
    }
    return (dstValues, dstTimestamps);
}

var (
rollupResultCacheFullHits    = metrics.NewCounter(`vm_rollup_result_cache_full_hits_total`)
rollupResultCachePartialHits = metrics.NewCounter(`vm_rollup_result_cache_partial_hits_total`)
rollupResultCacheMiss        = metrics.NewCounter(`vm_rollup_result_cache_miss_total`)
)

fn evalRollupFuncWithMetricExpr(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: Expression,
    me: &MetricExpr,
    iafc: IncrementalAggrFuncContext,
    windowExpr: &DurationExpr) -> Result<Vec<Timeseries>, Error> {
    let mut rollupMemorySize: i64;

    let window = windowExpr.Duration(ec.Step);
    qt = qt.NewChild("rollup %s(): timeRange=[%d..%d], step=%d, window=%d", func_name, ec.Start, ec.End, ec.Step, window)

    defer
    func()
    {
        qt.Donef("neededMemoryBytes=%d", rollupMemorySize)
    }
    ()
    if me.IsEmpty() {
        return evalNumber(ec, nan);
    }

    // Search for partial results in cache.
    let (tssCached, start) = rollupResultCacheV.Get(qt, ec, expr, window);
    if start > ec.End {
        // The result is fully cached.
        rollupResultCacheFullHits.Inc()
        return tssCached;
    }
    if start > ec.Start {
        rollupResultCachePartialHits.Inc()
    } else {
        rollupResultCacheMiss.Inc()
    }

    // Obtain rollup configs before fetching data from db,
    // so type errors can be caught earlier.
    let sharedTimestamps = getTimestamps(start, ec.end, ec.step);
    let (preFunc, rcs) = getRollupConfigs(func_name, rf, expr, start, ec.end, ec.step, window, ec.lookbackDelta, sharedTimestamps)?;

    // Fetch the remaining part of the result.
    let tfs = searchutils.ToTagFilters(me.LabelFilters);
    let tfss = searchutils.JoinTagFilterss([][]storage.TagFilter{ tfs }, ec.EnforcedTagFilterss)
    let minTimestamp = start - maxSilenceInterval;
    if window > ec.Step {
        minTimestamp -= window
    } else {
        minTimestamp -= ec.Step
    }
    let sq = storage.NewSearchQuery(minTimestamp, ec.end, tfss, ec.maxSeries);
    let rss = netstorage.ProcessSearchQuery(qt, sq, true, ec.Deadline)?;
    let rssLen = rss.Len();
    if rssLen == 0 {
        rss.cancel();
        tss = mergeTimeseries(tssCached, None, start, ec);
        return tss;
    }

    // Verify timeseries fit available memory after the rollup.
    // Take into account points from tssCached.
    let pointsPerTimeseries = 1 + (ec.end - ec.start) / ec.step;
    let mut timeseriesLen = rssLen;
    if iafc != nil {
        // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
        timeseriesLen = cgroup.AvailableCPUs();
        if iafc.ae.Modifier.Op != "" {
            if iafc.ae.Limit > 0 {
                // There is an explicit limit on the number of output time series.
                timeseriesLen *= iafc.ae.Limit
            } else {
                // Increase the number of timeseries for non-empty group list: `aggr() by (something)`,
                // since each group can have own set of time series in memory.
                timeseriesLen *= 1000
            }
        }
        // The maximum number of output time series is limited by rssLen.
        if timeseriesLen > rssLen {
            timeseriesLen = rssLen
        }
    }
    let rollupPoints = mulNoOverflow(pointsPerTimeseries, int64(timeseriesLen * len(rcs)))
    let rollupMemorySize = mulNoOverflow(rollupPoints, 16);
    let rml = getRollupMemoryLimiter();
    if !rml.Get(uint64(rollupMemorySize)) {
        rss.Cancel()
        return nil;, fmt.Errorf("not enough memory for processing %d data points across %d time series with %d points in each time series; " +
                                    "total available memory for concurrent requests: %d bytes; " +
                                    "requested memory: %d bytes; " +
                                    "possible solutions are: reducing the number of matching time series; switching to node with more RAM; " +
                                    "increasing -memory.allowedPercent; increasing `step` query arg (%gs)",
                                rollupPoints, timeseriesLen * len(rcs), pointsPerTimeseries, rml.MaxSize, uint64(rollupMemorySize), float64(ec.Step) / 1e3)
    }
    defer
    rml.Put(uint64(rollupMemorySize));

// Evaluate rollup
    let keepMetricNames = getKeepMetricNames(expr);
    let mut tss: &Vec<Timeseries>;
    if iafc != nil {
        tss = evalRollupWithIncrementalAggregate(qt, func_name, keepMetricNames, iafc, rss, rcs, preFunc, sharedTimestamps)?;
    } else {
        tss = evalRollupNoIncrementalAggregate(qt, func_name, keepMetricNames, rss, rcs, preFunc, sharedTimestamps)?;
    }

    tss = mergeTimeseries(tssCached, tss, start, ec);
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
                                      iafc: IncrementalAggrFuncContext, rss: Results,
                                      rcs: &[RollupConfig],
                                      preFunc: fn(values: Vec<f64>, timestamps: Vec<i64>),
                                      sharedTimestamps: Vec<i64>) -> Result<Vec<Timeseries>, Error> {
    let mut qt_ = qt.NewChild("rollup %s() with incremental aggregation %s() over %d series", func_name, iafc.ae.Name, rss.Len())
    // defer qt.Done()

    RunParallel(qt, |rs: Result, workerID: usize| {
        rs.Values,
        rs.Timestamps = dropStaleNaNs(func_name, rs.values, rs.timestamps)
        preFunc(rs.calues, rs.timestamps)
        let ts = getTimeseries()
        defer
        putTimeseries(ts)
        for rc in rcs {
            if tsm: = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.MetricName);
            if Some(tsm) {
                rc.DoTimeseriesMap(tsm, rs.Values, rs.Timestamps)
                for ts in tsm.m {
                    iafc.updateTimeseries(ts, workerID)
                }
                continue;
            }
            ts.Reset()
            doRollupForTimeseries(funcName, keepMetricNames, rc, ts, &rs.MetricName, rs.Values, rs.Timestamps, sharedTimestamps)
            iafc.updateTimeseries(ts, workerID)

// ts.Timestamps points to sharedTimestamps. Zero it, so it can be re-used.
            ts.Timestamps = nil
            ts.denyReuse = false
        }
        return nil;
    })

    let tss = iafc.finalizeTimeseries();
    qt.Printf("series after aggregation with %s(): %d", iafc.ae.Name, len(tss))
    return tss;
}

fn evalRollupNoIncrementalAggregate(
    qt: &QueryTracer,
    funcName: String,
    keepMetricNames: bool,
    rss: netstorage.Results,
    rcs: Vec<RollupConfig>,
    preFunc: fn(values: Vec<f64>, timestamps: Vec<i64>, sharedTimestamps: Vec<i64>),
) -> Result<Vec<Timeseries>, Error> {
    let qt = qt.NewChild("rollup %s() over %d series", funcName, rss.len());

    defer
    qt.Done()

    let tss: Vec<Timeseries> = Vec: with_capacity(rss.len() * rcs.len());

    let tssLock: sync.Mutex
    rss.RunParallel(qt, move |rs: &Result, workerID: uint | {
        (rs.values, rs.timestamps) = dropStaleNaNs(funcName, rs.Values, rs.Timestamps)
        preFunc(rs.values, rs.timestamps);
        for rc in rcs {
            let tsm = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.MetricName);
            if Some(tsm) {
                rc.doTimeseriesMap(tsm, rs.Values, rs.Timestamps);
                tssLock.Lock();
                tss = tsm.AppendTimeseriesTo(tss)
                tssLock.Unlock();
                continue;
            }
            let ts: Rimeseries;
            doRollupForTimeseries(funcName, keepMetricNames, rc, &ts, &rs.MetricName, rs.Values, rs.Timestamps, sharedTimestamps)
            tssLock.Lock();
            tss.push(ts);
            tssLock.Unlock()
        }
    });

    return tss;
}

fn doRollupForTimeseries(funcName: String, keep_metric_names: bool, rc: RollupConfig, tsDst: &Timeseries, mnSrc: &MetricName,
                         valuesSrc: Vec<f64>, timestampsSrc: Vec<i64>, sharedTimestamps: Vec<i64>) {
    tsDst.metric_name.copy_from(mnSrc);
    if len(rc.TagValue) > 0 {
        tsDst.metric_name.add_tag("rollup", rc.TagValue)
    }
    if !keep_metric_names && !rollupFuncsKeepMetricName[funcName] {
        tsDst.MetricName.ResetMetricGroup()
    }
    tsDst.values = rc.Do(tsDst.Values[: 0], valuesSrc, timestampsSrc)
    tsDst.Timestamps = sharedTimestamps
    tsDst.denyReuse = true
}

var bbPool bytesutil.ByteBufferPool

fn evalNumber(ec: &EvalConfig, n: f64) -> Vec<Timeseries> {
    let timestamps = ec.getSharedTimestamps();
    let mut values = Vec::with_capacity(timestamps.len());
    for i in 0..timestamps.len() {
        values[i] = n;
    }
    let ts = Timeseries::with_shared_timestamps(timestamps, values);
    return vec![ts];
}

fn evalString(ec: &EvalConfig, s: String) -> Vec<Timeseries> {
    let mut rv = evalNumber(ec, nan);
    rv[0].metric_name.metric_group = s;
    return rv;
}

fn evalTime(ec: &EvalConfig) -> Vec<Timeseries> {
    let rv = evalNumber(ec, nan);
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

fn dropStaleNaNs(funcName: String, values: Vec<f64>, timestamps: &Vec<i64>) -> (Vec<f64>, Vec<i64>) {
    if *noStaleMarkers || funcName == "default_rollup" || funcName == "stale_samples_over_time" {
        // Do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // Do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return (values, timestamps);
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let hasStaleSamples = values.iter().any(|x| decimal.isStaleNaN(x));

    if !hasStaleSamples {
        // Fast path: values have no Prometheus staleness marks.
        return (values, timestamps);
    }
    // Slow path: drop Prometheus staleness marks from values.
    dstValues: = values
    [: 0]
    dstTimestamps = timestamps
    [: 0]
    for (i, v) in values.iter().enumerate() {
        if decimal.IsStaleNaN(v) {
            continue;
        }
        dstValues.push(v);
        dstTimestamps.push(timestamps[i]);
    }
    return (dstValues, dstTimestamps);
}