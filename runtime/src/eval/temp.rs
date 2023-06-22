use std::sync::Arc;
use std::task::Context;

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i32 = 50;


// QueryStats contains various stats for the query.
pub struct QueryStats {
    // series_fetched contains the number of series fetched from storage during the query evaluation.
    pub series_fetched: usize
}

impl QueryStats {
    pub fn add_series_fetched(&mut self, n: usize) {
        self.series_fetched += n;
    }
}


pub fn eval_expr(ctx: &Arc<Context>, ec: &EvalConfig, e: &Expr) -> RuntimeResult<Vec<Timeseries>>  {
    let is_tracing = ctx.trace_enabled();
    if is_tracing {
        let query = string(e.AppendString(None));
        query = bytesutil.LimitStringLen(query, 300);
        let may_cache = ec.mayCache();
        trace_span!("eval: query={}, timeRange={}, step={}, may_cache={}", query, ec.timeRangeString(), ec.step, may_cache)
    }
    let rv = eval_expr_internal(qt, ec, e)?;
    if is_tracing {
        let series_count = rv.len();
        let mut points_per_series = 0;
        if rv.is_empty() {
            points_per_series = rv[0].len();
        }
        let points_count = series_count * points_per_series;
        qt.Donef("series={}, points={}, points_per_series={}", series_count, points_count, points_per_series)
    }
    return Ok(rv)
}

fn eval_expr_internal(qt: &Tracer, ec: &Arc<EvalConfig>, e: Expr) -> RuntimeResult<Vec<Timeseries>> {
    match e {
        Expr::NumberExpr(n) => {
            eval_number(ec, n)
        },
        Expr::DurationExpr(de) => {
            let d = de.duration(ec.step);
            let d_sec = d / 1000_f64;
            eval_number(ec, d_sec)
        },
        Expr::StringExpr(se) => {
            eval_string(ec, se)
        }
        Expr::MetricExpression(me) => {
            let re = RollupExpr{
                Expr: me,
            };
            eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", me, err))?;
        },
        Expr::RollupExpr(re) => {
            eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", me, err))?;
        }
        Expr::FuncExpr(fe) => {
            let nrf = getRollupFunc(fe.Name);
            if nrf == None {
                trace_span!("transform {}()", fe.Name);
                let rv = eval_transform_func(qtChild, ec, fe)?;
                qtChild.Donef("series={}", rv.len());
                Ok(rv)
            };
            let (args, re) = eval_rollup_func_args(qt, ec, &fe)?;
            let rf = nrf(args)?;
            eval_rollup_func(qt, ec, fe.Name, rf, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", fe, err))
        }
        Expr::AggrFuncExpr(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(qtChild, ec, ae)?;
            trace!("series={}", rv.len());
            Ok(rv)
        },
        Expr::BinaryExpr(be) => {
            trace_span!("binary op {}", be.op);
            let rv = eval_binary_op(ctx, ec, be)?;
            trace!("series={}", rv.len());
            Ok(rv)
        },
        _ => {
            return Err(RuntimeError::From(format!("unexpected expression {}", e)))
        }
    }
}

fn eval_transform_func(ctx: &Arc<Context>, ec: &EvalConfig, fe: &FuncExpr) -> RuntimeResult<Vec<Timeseries>> {
    let (tf, args) = match fe.function {
        BuiltInFunction::Transform(func) => {
            let handler = get_transform_func(func);
            let args = match fe.function {
                TransformFunction::Union() => {
                    eval_exprs_in_parallel(ctx, ec, fe.args)?
                },
                _ => {
                    eval_exprs_sequentially(ctx, ec, fe.args)?
                }
            };
            (handler, args)
        }
        _ => {
            Err(RuntimeError::InvalidFunction(format!("unknown func {}", fe.function.name())))
        }
    }?;
    let tfa = TransformFuncArg{
        ec,
        fe,
        args
    };
    tf(tfa).map_error(|err| format!("cannot evaluate {}: {}", fe, err))
}

fn eval_aggr_func(ctx: &Arc<Context>, ec: &Arc<EvalConfig>, ae: &AggrFuncExpr) -> RuntimeResult<Vec<Timeseries>> {
    if let Some(callbacks) = getIncrementalAggrFuncCallbacks(ae.Name) {
        let (fe, nrf) = tryGetArgRollupFuncWithMetricExpr(ae);
        if let Some(fe) = fe {
            // There is an optimized path for calculating AggrFuncExpr over rollupFunc over MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = evalRollupFuncArgs(qt, ec, fe)?;
            let rf = nrf(args);
            let iafc = newIncrementalAggrFuncContext(ae, callbacks);
            return eval_rollup_func(ctx, ec, fe.Name, rf, ae, re, iafc)
        }
    }
    let args = evalExprsInParallel(qt, ec, ae.Args)?;
    let af = getAggrFunc(ae.Name);
    let afa = AggrFuncArg{ ae, args, ec};

    qtChild := qt.NewChild("eval {}", ae.Name)

    af(afa)
        .map_error(|err| format!("cannot evaluate {}: {}", ae, err))
}

fn map_err_handler(e: Error) -> RuntimeError
    let msg = "cannot execute {}: {}", be, err)
}

fn eval_binary_op(ctx: &Arc<Context>, ec: &Arc<EvalConfig>, be: &BinaryOpExpr) -> RuntimeResult<Vec<Timeseries>> {
    let bf = getBinaryOpFunc(be.Op);
    let mut tss_left: Vec<Timeseries>;
    let mut tss_right: Vec<Timeseries>;
    
    match be.Op {
        And | If => {
            // Fetch right-side series at first, since it usually contains
            // lower number of time series for `and` and `if` operator.
            // This should produce more specific label filters for the left side of the query.
            // This, in turn, should reduce the time to select series for the left side of the query.
            (tss_right, tss_left) = exec_binary_op_args(ec, be.Right, be.Left, be)?;
        }
        _ => {
            (tss_left, tss_right) = exec_binary_op_args(ec, be.Left, be.Right, be)?;            
        }
    }
if err != None {

}
let mut bfa = BinaryOpFuncArg{
    be: &be,
    left:  &tss_left,
    right: &tss_right,
};
    bf(bfa).map_err(|err| format!("cannot evaluate {}: {}", be, err))
}


fn exec_binary_op_args(ctx: &Arc<Context>,
                       ec: &Arc<EvalConfig>,
                       expr_first: &Expr,
                       expr_second: &Expr,
                       be: &BinaryOpExpr) -> RuntimeResult((Vec<Timeseries>, Vec<Timeseries>)) {
    if !canPushdownCommonFilters(be) {
        // Execute expr_first and expr_second in parallel, since it is impossible to pushdown common filters
        // from expr_first to expr_second.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886
        trace_span!("execute left and right sides of {} in parallel", be.Op);

        trace!("expr1");
        go func() {
            tssFirst, errFirst = evalExpr(qtFirst, ec, expr_first)
        }()

var tssSecond []*timeseries
qtSecond := qt.NewChild("expr2")
go func() {
tssSecond, errSecond = evalExpr(qtSecond, ec, expr_second)
}()

wg.Wait()
return tssFirst, tssSecond
}

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
let tss_first = eval_expr(qt, ec, expr_first)?;
if len(tss_first) == 0 && strings.ToLower(be.Op) != "or" {
// Fast path: there is no sense in executing the expr_second when expr_first returns an empty result,
// since the "expr_first op expr_second" would return an empty result in any case.
// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349
return None, None, None
}
    let mut lfs = getCommonLabelFilters(tss_first);
    let lfs = trim_filters_by_group_modifier(lfs, be);
    let expr_second = pushdown_binary_op_filters(expr_second, lfs);
    let tss_second = eval_expr(qt, ec, expr_second)?;
    return (tss_first, tss_second);
}


fn tryGetArgRollupFuncWithMetricExpr(ae: &AggrFuncExpr) (*metricsql.FuncExpr, newRollupFunc) {
if len(ae.Args) != 1 {
return None, None
}
e := ae.Args[0]
// Make sure e contains one of the following:
// - metricExpr
// - metricExpr[d]
// - rollupFunc(metricExpr)
// - rollupFunc(metricExpr[d])

if me, ok := e.(*metricsql.MetricExpr); ok {
// e = metricExpr
if me.IsEmpty() {
return None, None
}
let fe = FuncExpr::default_rollup(me);
nrf = getRollupFunc(fe.name)
return (fe, nrf)
}
if re, ok := e.(*metricsql.RollupExpr); ok {
if me, ok := re.Expr.(*metricsql.MetricExpr); !ok || me.IsEmpty() || re.ForSubquery() {
return None, None
}
// e = metricExpr[d]
fe := &metricsql.FuncExpr{
Name: "default_rollup",
Args: []metricsql.Expr{re},
}
nrf := getRollupFunc(fe.Name)
return fe, nrf
}
fe, ok := e.(*metricsql.FuncExpr)
if !ok {
return None, None
}
let nrf = getRollupFunc(fe.name)?;
let rollupArgIdx := metricsql.GetRollupArgIdx(fe)
if rollupArgIdx >= fe.args.len() {
// Incorrect number of args for rollup func.
return None, None
}
arg := fe.Args[rollupArgIdx]
if me, ok := arg.(*metricsql.MetricExpr); ok {
if me.IsEmpty() {
return None, None
}
// e = rollupFunc(metricExpr)
return FuncExpr{
Name: fe.Name,
Args: []metricsql.Expr{me},
}, nrf
}
if re, ok := arg.(*metricsql.RollupExpr); ok {
if me, ok := re.Expr.(*metricsql.MetricExpr); !ok || me.IsEmpty() || re.ForSubquery() {
return None, None
}
// e = rollupFunc(metricExpr[d])
return fe, nrf
}
return None, None
}

pub(super) fn eval_exprs_sequentially(ec: &EvalConfig, es: &[Expr]) -> RuntimeResult<Vec<Vec<Timeseries>>>) {
    let rvs: Vec<Vec<Timestamps>> = Vec:with_capacity(es.len());
    for e in es {
        let rv = eval_expr(qt, ec, e)?;
        rvs.push(rv)
    }
    Ok(rvs)
}

pub(super) fn eval_exprs_in_parallel(ec: &EvalConfig, es: &[Expr]) ([][]*timeseries, error) {
    if es.len() < 2 {
        return eval_exprs_sequentially(ec, es)
    }
    let rvs = Vec::with_capacity(es.len());
    trace!("eval function args in parallel")
    for e in es {
        qtChild := qt.NewChild("eval arg {}", i)
        go func(e metricsql.Expr, i int) {
            let rv = eval_expr(qtChild, ec, e)?;
            rvs.push(rv)
        }(e, i)
    }
    return rvs
}

pub(super) fn eval_rollup_func_args(
    ctx: &Arc<Context>,
    ec: &Arc<EvalConfig>,
    fe: &FuncExpr) -> RuntimeResult<(Vec<Value>, RollupExpr)> {
    let mut re: RollupExpr;

    let rollup_arg_idx = metricsql.GetRollupArgIdx(fe);
    if fe.args.len() <= rollup_arg_idx {
        let msg = format!("expecting at least {} args to {}; got {} args; expr: {}",
                          rollup_arg_idx +1, fe.Name, fe.args.len, fe);
        return Err(RuntimeResult::General(msg))
    }

    let args = Vec::with_capacity( fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = getRollupExprArg(arg);
            args.push(re);
            continue
        }
        let ts = eval_expr(ctx, ec, arg)
            .map_err(
                |err| Err(RuntimError::General(
                    format!("cannot evaluate arg #{} for {}: {}", i+1, fe, err)))
            )?;

        args.push(ts);
    }

    return (args, re)
}

fn get_rollup_expr_arg(arg: &Expr) -> RollupExpr {
    re, ok := arg.(*metricsql.RollupExpr);
    if !ok {
// Wrap non-rollup arg into metricsql.RollupExpr.
return &metricsql.RollupExpr{
Expr: arg,
}
}
if !re.for_subquery() {
// Return standard rollup if it doesn't contain subquery.
return re
}
me, ok := re.Expr.(*metricsql.MetricExpr)
if !ok {
// arg contains subquery.
return re
}
// Convert me[w:step] -> default_rollup(me)[w:step]
reNew := *re
reNew.Expr = &metricsql.FuncExpr{
Name: "default_rollup",
Args: []metricsql.Expr{
&metricsql.RollupExpr{Expr: me},
},
}
return &reNew
}

// expr may contain:
// - rollupFunc(m) if iafc is None
// - aggrFunc(rollupFunc(m)) if iafc isn't None
fn eval_rollup_func(
    ctx: &Arc<Context>,
    ec: &Arc<EvalConfig>,
    func_name: string,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: &IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {
    if re.at.is_none() {
        return eval_rollup_func_without_at(ctx, ec, func_name, rf, expr, re, iafc)
    }
    let tss_at = eval_expr(ctx, ec, re.at)
        .map_err(|err| UserReadableError::from_err(format!("cannot evaluate `@` modifier: {}", err)))?;

    if len(tss_at) != 1 {
        return None, &UserReadableError{
            Err: fmt.Errorf("`@` modifier must return a single series; it returns {} series instead", len(tss_at)),
        }
    }
    let at_timestamp = tss_at[0].values[0] * 1000_f64;
    let mut ec_new = copyEvalConfig(ec);
    ec_new.start = at_timestamp;
    ec_new.end = at_timestamp;
    let mut tss = eval_rollup_func_without_at(ctx, ec_new, func_name, rf, expr, re, iafc)?;

    // expand single-point tss to the original time range.
    let timestamps = ec.getSharedTimestamps();
    for ts in tss.iter_mut() {
        let v = ts.values[0];
        ts.timestamps = Arc::clone(timestamps);
        ts.values = vec![v; timestamps.len()]
    }
    Ok(tss)
}

pub fn eval_rollup_func_without_at(
    ctx: &Arc<Context>,
    ec: &Arc<EvalConfig>,
    func: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: &IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {

    let ec_new = ec;
    let mut offset: i64 = 0;
    if let Some(ofs) = re.Offset {
        offset = ofs.duration(ec.step);
        ec_new = copyEvalConfig(ec_new);
        ec_new.start -= offset;
        ec_new.end -= offset;
        // There is no need in calling adjust_start_end() on ec_new if ec_new.MayCache is set to true,
        // since the time range alignment has been already performed by the caller,
        // so cache hit rate should be quite good.
        // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
    }
    if func == RollupFunction::Candlestick {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        step = ec_new.step;
        ec_new = copyEvalConfig(ec_new);
        ec_new.start += step;
        ec_new.end += step;
        offset -= step
    }
    var rvs []*timeseries
if me, ok := re.Expr.(*metricsql.MetricExpr); ok {
    rvs = eval_rollup_func_with_metric_expr(ec_new, funcName, rf, expr, me, iafc, re.window)?;
} else {
    if iafc.is_some() {
        logger.Panicf("BUG: iafc must be None for rollup {} over subquery {}", funcName, re)
    }
    rvs = eval_rollup_func_with_subquery(ctx,ec_new, funcName, rf, expr, re)?;
}

if funcName == "absent_over_time" {
    rvs = aggregate_absent_over_time(ec, re.expr, rvs)
}
if offset != 0 && !rvs.is_empty() {
    // Make a copy of timestamps, since they may be used in other values.
    let dst_timestamps = *rvs[0].timestamps.clone();
    for ts in dst_timestamps.iter_mut() {
        *ts += offset
    }
    let timestamps = Arc::clone(dst_timestamps);
    for ts in rvs.iter_mut() {
        ts.timestamps = Arc::clone(timestamps);
    }
}
return rvs
}

// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
//
// Values for returned series are set to nan if at least a single tss series contains nan at that point.
// This means that tss contains a series with non-empty results at that point.
// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(ec: &EvalConfig, expr: &Expr, tss: &Vec<Timeseries>) -> RuntimeResult<Vec<Timeseries>> {
    let rvs = get_absent_timeseries(ec, expr)?;
    if tss.is_empty() {
        return rvs
    }
    for i in 0 .. tss[0].values {
        for ts in tss {
            if ts.values[i].is_nan() {
                rvs[0].Values[i] = f64::NAN;
                break
            }
        }
    }
    return rvs
}

fn eval_rollup_func_with_subquery(
      ctx: &Arc<Context>,
      ec: &EvalConfig,
      func: RollupFunction,
      rf: rollupFunc,
      expr: &Expr,
      re: &RollupExpr) -> RuntimeResult<Vec<Timeseries>> {
    // TODO: determine whether to use rollupResultCacheV here.
    trace!("subquery");
    let step = re.step.duration(ec.step);
    if step == 0 {
        step = ec.step
    }
    let window = re.window.duration(ec.step);

    let ec_sq = copyEvalConfig(ec);
    ec_sq.start -= window + maxSilenceInterval + step;
    ec_sq.end += step;
    ec_sq.step = step;
    ec_sq.max_points_per_series = *maxPointsSubqueryPerTimeseries;

    validatemax_points_per_series(ec_sq.start, ec_sq.end, ec_sq.step, ec_sq.max_points_per_series)
        .map_err(|err| format!("cannot evaluate subquery: {}", err))?;

    // unconditionally align start and end args to step for subquery as Prometheus does.
    (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);
    let tss_sq = eval_expr(qt, ec_sq, re.expr)?;
    if len(tss_sq) == 0 {
        return None, None
    }
    let shared_timestamps = get_timestamps(
        ec.start,
        ec.end,
        ec.step,
        ec.max_points_per_series
    )?;
    (preFunc, rcs) = getRollupConfigs(func_name, rf, expr, ec.start, ec.end, ec.step, ec.max_points_per_series, window, ec.LookbackDelta, shared_timestamps)?;

    let mut samples_scanned_total: u64;
    let keep_metric_names = getKeepMetricNames(expr);
    let tsw = getTimeseriesByWorkerID();
    let seriesByWorkerID := tsw.byWorkerID;
do_parallel(tss_sq, func(tsSQ *timeseries, values &[f64], timestamps &[i64], workerID: uint) (&[f64], &[i64]) {
values, timestamps = removeNanValues(values[:0], timestamps[:0], tsSQ.Values, tsSQ.Timestamps)
preFunc(values, timestamps)
    for rc in rcs {
        if let Some(tsm) = newTimeseriesMap(funcName: func_name, keep_metric_names,
                                            sharedTimestamps: shared_timestamps, &tsSQ.metric_name) {
            let samplesScanned = rc.DoTimeseriesMap(tsm, values, timestamps)
            samples_scanned_total, samplesScanned)
            seriesByWorkerID[workerID].tss = tsm.AppendTimeseriesTo(seriesByWorkerID[workerID].tss)
            continue
        }
        let ts = Timeseries::default();
        samplesScanned = do_rollup_for_timeseries(func_name, keep_metric_names, rc, &ts, &tsSQ.metric_name, values, timestamps, shared_timestamps)
atomic.AddUi64(&samples_scanned_total, samplesScanned)
seriesByWorkerID[workerID].tss = append(seriesByWorkerID[workerID].tss, &ts)
}
return values, timestamps
})
tss := make([]*timeseries, 0, len(tss_sq)*len(rcs))
for i := range seriesByWorkerID {
tss = append(tss, seriesByWorkerID[i].tss...)
}
putTimeseriesByWorkerID(tsw)

rowsScannedPerQuery.Update(float64(samples_scanned_total))
qt.Printf("rollup {}() over {} series returned by subquery: series={}, samplesScanned={}", funcName, len(tss_sq), len(tss), samples_scanned_total)
return tss, None
}


fn do_parallel(tss: Vec<Timeseries>,
               f: fn(ts: &Timeseries, values: &[f64], timestamps: &[i64], workerID: usize) -> (&[f64], &[i64])) {
    let mut workers = netstorage.MaxWorkers();
    if workers > tss.len() {
        workers = tss.len();
    }
    let series_per_worker = (tss.len() + workers - 1) / workers;
    let workChs = Vec::with_capacity(workers);
    for i in workChs {
        workChs[i] = make(chan *timeseries,
        seriesPerWorker: series_per_worker)
    }
    for (i, ts) in tss.iter().enumerate() {
        let idx = i % workChs.len();
        workChs[idx] <- ts
    }

    for i := 0; i < workers; i++ {
        go func(workerID uint) {
            var tmpValues &[f64]
            var tmpTimestamps &[i64]
            for ts := range workChs[workerID] {
                tmpValues, tmpTimestamps = f(ts, tmpValues, tmpTimestamps, workerID)
            }
        }(uint(i))
    }
}


fn eval_rollup_func_with_metric_expr(
     ctx: &Arc<Context>,
     ec: &Arc<EvalConfig>,
     func_name: string,
     rf: RollupFunc,
     expr: &Expr,
     me: &MetricExpr,
     iafc: &IncrementalAggrFuncContext,
     window_expr: &DurationExpr) -> RuntimeResult<Vec<Timeseries>> {
    var rollupMemorySize i64
    let window = window_expr.Duration(ec.step);
    if qt.Enabled() {
        trace_span!("rollup {}(): timeRange={}, step={}, window={}",
                         func_name, ec.timeRangeString(), ec.step, window);
        defer func() {
            qt.Donef("neededMemoryBytes={}", rollupMemorySize)
        }()
    }
    if me.is_empty() {
        return eval_number(ec, nan);
    }

    // Search for partial results in cache.
    let (tssCached, start) = rollupResultCacheV.Get(qt, ec, expr, window)?;
    if start > ec.end {
        // The result is fully cached.
        rollupResultCacheFullHits.Inc();
        return Ok(tssCached)
    }
    if start > ec.start {
        rollupResultCachePartialHits.Inc();
    } else {
        rollupResultCacheMiss.Inc()
    }

// Obtain rollup configs before fetching data from db,
// so type errors can be caught earlier.
let shared_timestamps = get_timestamps(start, ec.end, ec.step, ec.max_points_per_series);
let (pre_func, rcs) = getRollupConfigs(
    func_name,
    rf,
    expr,
    start,
    ec.end,
    ec.step,
    ec.max_points_per_series,
    window,
    ec.LookbackDelta, shared_timestamps);

// Fetch the remaining part of the result.
let tfs = to_tag_filters(me.LabelFilters);
let tfss = join_tag_filterss(tfs, ec.EnforcedTagFilterss);
    let mut min_timestamp = start - maxSilenceInterval;
    if window > ec.step {
        min_timestamp -= window
    } else {
        min_timestamp -= ec.step
    }
    let sq = storage.NewSearchQuery(min_timestamp, ec.end, tfss, ec.MaxSeries)
    let mut rss = netstorage.ProcessSearchQuery(qt, sq, ec.Deadline)?;
    let rss_len = rss.len();
    if rss_len == 0 {
        rss.cancel();
        mergeTimeseries(tssCached, None, start, ec);
    }
    ec.query_stats.addSeriesFetched(rss_len);

    // Verify timeseries fit available memory after the rollup.
    // Take into account points from tssCached.
    let points_per_timeseries = 1 + (ec.end-ec.start)/ec.step;
    let mut timeseries_len = rss_len;
    if let Some(iafc) = iafc {
        // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
        timeseries_len = cgroup.AvailableCPUs();
        if iafc.ae.Modifier.Op != "" {
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
let mut rollup_points = mulNoOverflow(points_per_timeseries, timeseries_len *len(rcs));
rollupMemorySize = sumNoOverflow(mulNoOverflow(i64(rss_len), 1000), mulNoOverflow(rollup_points, 16))
if maxMemory := i64(logQueryMemoryUsage.N); maxMemory > 0 && rollupMemorySize > maxMemory {
requestURI = ec.GetRequestURI()
logger.Warnf("remoteAddr={}, requestURI={}: the {} requires {} bytes of memory for processing; "+
"logging this query, since it exceeds the -search.logQueryMemoryUsage={}; "+
"the query selects {} time series and generates {} points across all the time series; try reducing the number of selected time series",
ec.QuotedRemoteAddr, requestURI, expr, rollupMemorySize, maxMemory,
        timeseriesLen: timeseries_len*len(rcs),
        rollupPoints: rollup_points)
}
if maxMemory := i64(maxMemoryPerQuery.N); maxMemory > 0 && rollupMemorySize > maxMemory {
rss.cancel()
return None, &UserReadableError{
Err: fmt.Errorf("not enough memory for processing {}, which returns {} data points across {} time series with {} points in each time series "+
"according to -search.maxMemoryPerQuery={}; requested memory: {} bytes; "+
"possible solutions are: reducing the number of matching time series; increasing `step` query arg (step=%gs); "+
"increasing -search.maxMemoryPerQuery",
                expr, rollup_points, timeseries_len *len(rcs), points_per_timeseries, maxMemory, rollupMemorySize, float64(ec.step)/1e3),
}
}
rml := getRollupMemoryLimiter()
if !rml.Get(u64(rollupMemorySize)) {
    rss.cancel()
return None, &UserReadableError{
Err: format!("not enough memory for processing {}, which returns {} data points across {} time series with {} points in each time series; "+
"total available memory for concurrent requests: {} bytes; "+
"requested memory: {} bytes; "+
"possible solutions are: reducing the number of matching time series; increasing `step` query arg (step=%gs); "+
"switching to node with more RAM; increasing -memory.allowedPercent",
expr, rollupPoints, timeseriesLen*len(rcs), pointsPerTimeseries, rml.MaxSize, ui64(rollupMemorySize), float64(ec.step)/1e3),
}
}
defer rml.Put(ui64(rollupMemorySize))

// Evaluate rollup
let keepMetricNames = getKeepMetricNames(expr)
var tss []*timeseries
let tss = if iafc.is_some() {
    evalRollupWithIncrementalAggregate(ctx, funcName, keepMetricNames, iafc, rss, rcs, preFunc, sharedTimestamps)
} else {
    evalRollupNoIncrementalAggregate(ctx, funcName, keepMetricNames, rss, rcs, preFunc, sharedTimestamps)
}
let tss = mergeTimeseries(tssCached, tss, start, ec)
rollupResultCacheV.Put(qt, ec, expr, window, tss)
return Ok(tss);
}


fn eval_rollup_with_incremental_aggregate(
    func_name: string,
    keep_metric_names: bool,
    iafc: &IncrementalAggrFuncContext,
    rss: &Results,
    rcs: Vec<RollupConfig>,
    pre_func: PreFunc,
    shared_timestamps: Arc<Vec<i64>>) -> RuntimeResult<Vec<Timeseries>> {
trace_span!("rollup {}() with incremental aggregation {}() over {} series; rollupConfigs={}", funcName, iafc.ae.Name, rss.Len(), rcs)

    let mut samples_scanned_total: u64;
    err := rss.RunParallel(qt, func(rs *netstorage.Result, workerID uint) error {
        (rs.values, rs.timestamps) = dropStaleNaNs(funcName, rs.values, rs.timestamps);
        preFunc: pre_func(rs.values, rs.timestamps)
    let ts = getTimeseries();
    for rc in rcs {
        if let Some(tsm) = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.metric_name) {
            let samplesScanned = rc.DoTimeseriesMap(tsm, rs.values, rs.timestamps)
            for ts in tsm.m {
                iafc.updateTimeseries(ts, workerID)
            }
            atomic.AddUi64(&samplesScannedTotal, samplesScanned)
            continue
        }
        ts.reset();
        let samplesScanned = doRollupForTimeseries(funcName, keepMetricNames, rc, ts, &rs.metric_name, rs.Values, rs.Timestamps, sharedTimestamps)
        atomic.AddUi64(&samplesScannedTotal, samplesScanned)
        iafc.updateTimeseries(ts, workerID)
    }
    return None
})
let tss = iafc.finalizeTimeseries();
rowsScannedPerQuery.Update(float64(samples_scanned_total))
qt.Printf("series after aggregation with {}(): {}; samplesScanned={}",
          iafc.ae.name,
          tss.len(),
          samples_scanned_total);
return tss
}

fn eval_rollup_no_incremental_aggregate(
    ctx: &Arc<Context>,
    func_name: string,
    keep_metric_names: bool,
    rss: &Results,
    rcs: &Vec<RollupConfig>,
    pre_func: PreFunc,
    shared_timestamps: Arc<Vec<i64>>) -> RuntimeResult<Vec<Timeseries>> {

    trace_span!("rollup {}() over {} series; rollupConfigs={}", func_name, rss.len(), rcs);

    let mut samples_scanned_total: usize;
    let tsw = getTimeseriesByWorkerID();
    let series_by_worker_id = tsw.byWorkerID;
    let series_len = rss.len();
    err := rss.RunParallel(qt, func(rs *netstorage.Result, workerID uint) error {
        (rs.values, rs.timestamps) = dropStaleNaNs(func_name, rs.values, rs.timestamps);
        preFunc: pre_func(rs.values, rs.timestamps)
    for rc in rcs {
        if let Some(tsm) = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.metric_name) {
            let samplesScanned = rc.doTimeseriesMap(tsm, rs.values, rs.timestamps)
            atomic.AddUi64(&samplesScannedTotal, samplesScanned)
            seriesByWorkerID[workerID].tss = tsm.AppendTimeseriesTo(seriesByWorkerID[workerID].tss)
            continue
        }
        let mut ts = Timeseries::default();
        let samplesScanned = doRollupForTimeseries(funcName, keepMetricNames, rc, &ts, &rs.metric_name, rs.Values, rs.Timestamps, sharedTimestamps)
        atomic.AddUi64(&samplesScannedTotal, samplesScanned)
        seriesByWorkerID[workerID].tss = append(seriesByWorkerID[workerID].tss, &ts)
    }
    return None
})
let mut tss := make([]*timeseries, 0, series_len *len(rcs))
for i := range seriesByWorkerID {
tss = append(tss,
        series_by_worker_id[i].tss...)
}

rowsScannedPerQuery.Update(float64(samples_scanned_total));
qt.Printf("samplesScanned={}", samples_scanned_total);
return tss
}

fn do_rollup_for_timeseries(func_name: &str,
                            keep_metric_names: bool,
                            rc: &RollupConfig,
                            ts_dst: &mut Timeseries,
                            mn_src: &metric_name,
                            values_src: &[f64],
                            timestamps_src: &[i64],
                            shared_timestamps: &[i64]) -> i64 {

    ts_dst.metric_name.CopyFrom(mn_src);
    if len(rc.tag_value) > 0 {
        ts_dst.metric_name.add_tag("rollup", rc.tag_value);
    }
    if !keep_metric_names && !rollupFuncsKeepMetricName[func_name] {
        ts_dst.metric_name.reset_metric_group();
    }
    let (values, samplesScanned) = rc.Do(&mut ts_dst, values_src, timestamps_src);
    ts_dst.timestamps = shared_timestamps;
    ts_dst.values = values;
    return samplesScanned
}