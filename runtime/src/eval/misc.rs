

// The minimum number of points per timeseries for enabling time rounding.
// This improves cache hit ratio for frequently requested queries over
// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i32 = 50;


fn eval_expr(qt: &QueryTracer, ec: EvalConfig, e: &Expression)-> RuntimeResult<Vec<Timeseries>> {
    if qt.Enabled() {
        let mut query = string(e.AppendString(nil));
        query = bytesutil.LimitStringLen(query, 300);
        let may_cache = ec.mayCache();
        qt = qt.NewChild("eval: query=%s, timeRange=%s, step=%d, may_cache=%v", query, ec.timeRangeString(), ec.Step, may_cache)
    }
    let rv = eval_expr_internal(qt, ec, e);
    if qt.Enabled() {
        let series_count = rv.len();
        let mut points_per_series = 0;
        if rv.len() > 0 {
            points_per_series = rv[0].timestamps.len();
        }
        let points_count = series_count * points_per_series;
        qt.Donef("series=%d, points=%d, points_per_series=%d", series_count, points_count, points_per_series)
    }
    rv
}

fn eval_expr_internal(qt: &QueryTracer, ec: EvalConfig, e: Expression) -> RuntimeResult<Vec<Timeseries>> {
    match e {
        Expression::MetricExpr(me) => {
            let re = &RollupExpr::new(me);
            match eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, nil) {
                Err(err) => {
                    let msg = format!("cannot evaluate {}: {}", me, err);
                },
                Ok(rv) => Ok(rv)
            }
        }
        Expression::Rollup(re) => {
            match eval_rollup_func(qt, ec, "default_rollup", rollupDefault, e, re, nil) {
                Err(e) => {
                    let msg = format!(`cannot evaluate {}: {}`, re, err);
                    return Err(RuntimeError::General(msg))
                },
                Ok(v) => Ok(v)
            }
        }
        Expression::FuncExpr(fe) => {
            nrf = getRollupFunc(fe.Name);
            if nrf == nil {
                return eval_transform_func(qtChild, ec, fe);
            }
            let (args, re) = eval_rollup_func_args(qt, ec, fe)?;
            rf = nrf(args)?;
            let rv = eval_rollup_func(qt, ec, fe.name, rf, e, re, nil);
            if err != nil {
                return nil, fmt.Errorf(`cannot evaluate %q: %w`, fe.AppendString(nil), err)
            }
            return rv
        }
        Expression::AggrExpr(ae) => {
            eval_aggr_func( ec, ae)
        }
        Expression::BinaryOpExpr(op) => {
            eval_binary_op(qtChild, ec, be)
        }
        Expression::DurationExpr(de) => {
            let d = de.duration(ec.step);
            let d_sec = d / 1000;
            return eval_number(ec, d_sec);
        }
        Expression::Number(num) => {
            return evalNumber(ec, ne.N)?
        }
        Expression::StringExpr(str) => {
            return eval_String(ec, se.S)
        }
        _ => {
            let msg = format!("unexpected expression {}", e)         ;
        }
    }
}


fn eval_transform_func(ec: &EvalConfig, fe: &FuncExpr) -> RuntimeResult<Vec<Timeseries>> {
    let tf = getTransformFunc(fe.name)?;
    let args: Vec<Vec<Timeseries>>;

    switch fe.name() {
        case "", "union":
            args = eval_exprs_in_parallel(ec, fe.args);
        default:
            args = eval_exprs_sequentially(ec, fe.args)?;
    }
    let tfa = TransformFuncArg{
        ec,
        fe:   &fe,
        args,
        }
    let rv = tf(tfa)?;
}

fn eval_aggr_func(ec: EvalConfig, ae: &AggrFuncExpr) -> RuntimeResult<Vec<Timeseries>> {
    if let Some(callbacks) = getIncrementalAggrFuncCallbacks(ae.name) {
        let (fe, nrf) = tryGetArgRollupFuncWithMetricExpr(ae);
        if fe != nil {
            // There is an optimized path for calculating AggrFuncExpr over rollupfn over MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = eval_rollup_func_args(qt, ec, fe)?;
            let rf = nrf(args);
            let iafc = IncrementalAggrFuncContext::new(ae, callbacks);
            return eval_rollup_func(qt, ec, fe.name, rf, ae, re, iafc);
        }
    }
    let args = eval_exprs_in_parallel(qt, ec, ae.Args)?;
    let af = getAggrFunc(ae.Name)?;
    let afa = AggrFuncArg { ae, args, ec };
    match af(afa) {
        Err(err) => {
            let msg = format!("cannot evaluate {}: {}", ae, err)
            return Err( RumtimeError::General(msg) )
        },
        Ok(rv) => Ok(rv)
    }
}

fn eval_binary_op(qt: &QueryTracer, ec: EvalConfig, be: &BinaryOpExpr) -> RuntimeResult<Vec<Timeseries>> {
    let bf = getBinaryOpFunc(be.Op)?;
    let err: RuntimeError;
    let tssLeft: Vec<Timeseries>;
    let tssRight: Vec<Timeseries>;

    switch strings.ToLower(be.Op) {
case "and", "if":
// Fetch right-side series at first, since it usually contains
// lower number of time series for `and` and `if` operator.
// This should produce more specific label filters for the left side of the query.
// This, in turn, should reduce the time to select series for the left side of the query.
            (tssRight, tssLeft) = exec_binary_op_args(qt, ec, be.Right, be.Left, be)?
default:
    (tssLeft, tssRight) = exec_binary_op_args(qt, ec, be.left, be.right, be)
}
if err != nil {
return nil, fmt.Errorf("cannot execute %q: %w", be.AppendString(nil), err)
}
let bfa = &BinaryOpFuncArg{
be:    be,
left:  tssLeft,
right: tssRight,
}
    let rv  = bf(bfa);
if err != nil {
return nil, fmt.Errorf(`cannot evaluate %q: %w`, be.AppendString(nil), err)
}
return rv
}

fn is_aggr_func_without_grouping(e: Expression) -> bool {
    match e {
        AggrFuncExpr(expr) => {
            expr.modifier.args.len() = 0
        },
        _ => false
    }
}

fn exec_binary_op_args(qt: &QueryTracer,
                       ec: &EvalConfig,
                       expr_first: &Expression,
                       expr_second: &Expression,
                       be: &BinaryOpExpr) -> RuntimeResult<(Vec<timeseries>, Vec<Timeseries>)> {

    if !canPushdownCommonFilters(be) {
        // Execute exprFirst and expr_second in parallel, since it is impossible to pushdown common filters
        // from exprFirst to expr_second.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886
        var tssFirst []*timeseries
        var errFirst error
        go func() {
            tssFirst = evalExpr(qtFirst, ec, exprFirst)?;
        }()

var tssSecond []*timeseries
go func() {
defer wg.Done()
tssSecond, errSecond = evalExpr(qtSecond, ec, exprSecond);
qtSecond.Done()
}()
if errFirst != nil {
return nil, nil
}
if errSecond != nil {
return nil, nil, errSecond
}
    return (tssFirst, tssSecond)
}

// Execute binary operation in the following way:
//
// 1) execute the exprFirst
// 2) get common label filters for series returned at step 1
// 3) push down the found common label filters to expr_second. This filters out unneeded series
//    during expr_second exection instead of spending compute resources on extracting and processing these series
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
let tssFirst = eval_expr(ec, exprFirst)?;
let lfs = getCommonLabelFilters(tssFirst);
lfs = TrimFiltersByGroupModifier(lfs, be);
let expr_second = PushdownBinaryOpFilters(exprSecond, lfs);
let tssSecond = eval_expr(ec, expr_second)?;
return (tssFirst, tssSecond)
}


fn tryGetArgRollupFuncWithMetricExpr(ae: &AggrFuncExpr) -> (&FuncExpr, newRollupFunc) {
if ae.args.len() != 1 {
return nil, nil
}
let e = ae.args[0];
// Make sure e contains one of the following:
// - metricExpr
// - metricExpr[d]
// - rollupFunc(metricExpr)
// - rollupFunc(metricExpr[d])

if me, ok = e.(*MetricExpr); ok {
// e = metricExpr
if me.IsEmpty() {
return nil, nil
}
fe = &FuncExpr{
Name: "default_rollup",
Args: []Expression{me},
}
nrf = getRollupFunc(fe.Name)
return fe, nrf
}
if re, ok = e.(*RollupExpr); ok {
if me, ok = re.Expr.(*MetricExpr); !ok || me.IsEmpty() || re.ForSubquery() {
return nil, nil
}
// e = metricExpr[d]
    fe = &FuncExpr{
        Name: "default_rollup",
        Args: []Expression{re},
    }
    nrf = getRollupFunc(fe.Name)?
    return (fe, nrf)
}
fe, ok = e.(*FuncExpr)
if !ok {
return nil, nil
}
nrf = getRollupFunc(fe.Name)
if nrf == nil {
return nil, nil
}
rollupArgIdx = GetRollupArgIdx(fe)
if rollupArgIdx >= len(fe.Args) {
// Incorrect number of args for rollup func.
return nil, nil
}
arg = fe.Args[rollupArgIdx]
if me, ok = arg.(*MetricExpr); ok {
if me.IsEmpty() {
return nil, nil
}
// e = rollupFunc(metricExpr)
return &FuncExpr{
Name: fe.Name,
Args: []Expression{me},
}, nrf
}
if re, ok = arg.(*RollupExpr); ok {
if me, ok = re.Expr.(*MetricExpr); !ok || me.IsEmpty() || re.ForSubquery() {
return nil, nil
}
// e = rollupFunc(metricExpr[d])
return fe, nrf
}
return nil, nil
}

fn eval_exprs_sequentially(qt: &QueryTracer, ec: EvalConfig, es: &[Expression]) -> RuntimeResult<Vec<Vec<Timeseries>>> {
    for e in es.iter() {
        let rv = eval_expr(qt, ec, e)?;
        rvs.push(rv);
    }
    return rvs
}

fn eval_exprs_in_parallel(ec: EvalConfig, es: &Vec<Expression>) -> RuntimeResult<Vec<Vec<Timeseries>>> {
    if len(es) < 2 {
        return eval_exprs_sequentially(ec, es);
    }
    let rvs = Vec::with_capacity(es.length);
    let errs: Vec<RuntimeError> = vec![];
    // todo: par_iter()
    for e in es.iter() {
        match eval_expr(ec, e) {
            Err(e) => {
                errs.push(e);
            },
            Ok(rv) => rvs.push(rv)
        }
    }
    if errs.len() {
        return Err(errs[0])
    }
    return rvs
}

fn eval_rollup_func_args(ec: EvalConfig, fe: &FuncExpr) ([]interface{}, *RollupExpr, error) {
    var re *RollupExpr
    let rollupArgIdx = GetRollupArgIdx(fe);
    if len(fe.Args) <= rollupArgIdx {
        return fmt.Errorf("expecting at least %d args to %q; got %d args; expr: %q", rollupArgIdx+1, fe.Name, len(fe.Args), fe.AppendString(nil))
    }
    let args = make([]interface{}, len(fe.Args))
    for (i, arg) in fe.args.iter() {
        if i == rollupArgIdx {
            re = getRollupExprArg(arg)
            args[i] = re
            continue
        }
        let ts = eval_expr(qt, ec, arg);
        if err != nil {
            return nil, nil, fmt.Errorf("cannot evaluate arg #%d for %q: %w", i+1, fe, err)
        }
        args[i] = ts
    }
    return args, re
}

fn getRollupExprArg(arg: &Expression) -> RollupExpr {
    match arg {
        Expression::Rollup(re) => {
            
        }
    }
re, ok = arg.(*RollupExpr)
if !ok {
// Wrap non-rollup arg into RollupExpr.
return &RollupExpr{
Expr: arg,
}
}
if !re.ForSubquery() {
// Return standard rollup if it doesn't contain subquery.
return re
}
me, ok = re.Expr.(*MetricExpr)
if !ok {
// arg contains subquery.
return re
}
// Convert me[w:step] -> default_rollup(me)[w:step]
reNew = *re
reNew.Expr = &FuncExpr{
Name: "default_rollup",
Args: []Expression{
&RollupExpr{Expr: me},
},
}
return &reNew
}

// expr may contain:
// - rollupFunc(m) if iafc is nil
// - aggrFunc(rollupFunc(m)) if iafc isn't nil
fn eval_rollup_func(ec: EvalConfig,
                    func_name: String,
                    rf: RollupFunc,
                    expr: Expression,
                    re: RollupExpr,
                    iafc: &mut IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {
    if re.At == nil {
        return eval_rollup_func_without_at(qt, ec, func_name, rf, expr, re, iafc);
    }
    tssAt = eval_expr(qt, ec, re.at);
    if err != nil {
        return nil, &UserReadableError{
        Err: fmt.Errorf("cannot evaluate `@` modifier: %w", err),
        }
    }
    if len(tssAt) != 1 {
        return nil, &UserReadableError{
        Err: fmt.Errorf("`@` modifier must return a single series; it returns %d series instead", len(tssAt)),
    }
}
    let at_timestamp = int64(tssAt[0].values[0] * 1000);
    let ec_new = copyEvalConfig(ec);
    ec_new.Start = at_timestamp;
    ec_new.End = at_timestamp;
    let tss = eval_rollup_func_without_at(qt, ec_new, func_name, rf, expr, re, iafc)?;
    // expand single-point tss to the original time range.
    let timestamps = ec.getSharedTimestamps();
    for ts in tss.iter() {
        let v = ts.values[0];
        let values = Vec::with_capacity(timestamps.len());
        for i in 0 .. timestamps.len() {
            values[i] = v
        }
        ts.timestamps = timestamps;
        ts.values = values
    }
    
    return tss
}

fn eval_rollup_func_without_at(
    qt: &QueryTracer,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: Expression,
    re: RollupExpr,
    iafc: &mut IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {
    func_name = strings.ToLower(funcName);
    let ecNew = ec;

    let offset: i64;
    if re.offset != nil {
        offset = re.offset.duration(ec.Step);
        ecNew = copyEvalConfig(ecNew);
        ecNew.Start -= offset;
        ecNew.End -= offset
    // There is no need in calling AdjustStartEnd() on ecNew if ecNew.MayCache is set to true,
    // since the time range alignment has been already performed by the caller,
    // so cache hit rate should be quite good.
    // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
    }
    if funcName == "rollup_candlestick" {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        let step = ecNew.Step;
        ecNew = copyEvalConfig(ecNew);
        ecNew.Start += step;
        ecNew.End += step;
        offset -= step
    }
    let rvs: Vec<Timeseries>;
    match re.expr {
        Expression::MetricExpression(me) => {
            rvs = eval_rollup_func_with_metric_expr(qt, ecNew, funcName, rf, expr, me, iafc, re.window);
        },
        _ => {
            if iafc != nil {
                logger.Panicf("BUG: iafc must be nil for rollup {} over subquery {}", funcName, re);
            }
            rvs = eval_rollup_func_with_subquery(ctx, ecNew, funcName, rf, expr, re)?
        }
    }

    if funcName == "absent_over_time" {
        rvs = aggregate_absent_over_time(ec, re.Expr, rvs)
    }
    if offset != 0 && rvs.len() > 0 {
        // Make a copy of timestamps, since they may be used in other values.
        let src_timestamps = rvs[0].timestamps;
        let dst_timestamps = src_timestamps.clone();
        for i in 0 .. dst_timestamps.len() {
            dst_timestamps[i] += offset
        }
        for ts in rvs.iter_mut() {
            ts.timestamps = Arc::clone(&dst_timestamps)
        }
    }

    return rvs
}

// aggregate_absent_over_time collapses tss to a single time series with 1 and nan values.
//
// Values for returned series are set to nan if at least a single tss series contains nan at that point.
// This means that tss contains a series with non-empty results at that point.
// This follows Prometheus logic - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2130
fn aggregate_absent_over_time(ec: EvalConfig, expr: Expression, tss: &Vec<Timeseries>) -> ReturnResult<Vec<Timeseries>> {
    let rvs = getAbsentTimeseries(ec, expr)?;
    if tss.len() == 0 {
        return rvs
    }
    for i in 0 .. tss[0].values.len() {
        for ts in tss.iter() {
            if ts.values[i].is_nan() {
                rvs[0].values[i] = f64::NAN;
                break
            }
        }
    }
    return rvs
}

fn eval_rollup_func_with_subquery(qt: &QueryTracer,
                                  ec: &EvalConfig,
                                  func_name: String,
                                  rf: RollupFunc,
                                  expr: &Expression,
                                  re: RollupExpr) -> RuntimeResult<Vec<Timeseries>> {

    // TODO: determine whether to use rollupResultCacheV here.

    let step = re.step.Duration(ec.step);
    if step == 0 {
        step = ec.Step
    }
    window = re.window.Duration(ec.Step);

    let ec_sq = copyEvalConfig(ec);
    ec_sq.Start -= window + maxSilenceInterval + step;
    ec_sq.End += step;
    ec_sq.Step = step;
    ec_sq.MaxPointsPerSeries = *maxPointsSubqueryPerTimeseries;
    validateMaxPointsPerSeries(ec_sq.Start, ec_sq.End, ec_sq.Step, ec_sq.MaxPointsPerSeries)?;

    // unconditionally align start and end args to step for subquery as Prometheus does.
    (ec_sq.start, ec_sq.end) = alignStartEnd(ec_sq.Start, ec_sq.End, ec_sq.Step);
    let tss_sq = evalExpr(qt, ec_sq, re.expr)?;
    if tss_sq.len() == 0 {
        return Ok(vec![]);
    }
    let shared_timestamps = getTimestamps(ec.Start, ec.End, ec.Step, ec.MaxPointsPerSeries);
    let (pre_func, rcs) = getRollupConfigs(
        funcName,
        rf,
        expr,
        ec.start,
        ec.end,
        ec.step,
        ec.max_points_per_series,
        window,
        ec.LookbackDelta,
        shared_timestamps);

    let tss = Vec::with_capacity(tss_sq.len*() * rcs.len());
    let tssLock: Mutex;
    let mut samples_scanned_total: i64 = 0;
let keep_metric_names = get_keep_metric_names(expr);
do_parallel(tss_sq, |ts_sq: &Timeseries, values: &[f64], timestamps: &[i64]| -> (&[f64], &[i64]) {
    (values, timestamps) = removeNanValues(values, timestamps, ts_sq.values, ts_sq.timestamps);
    pre_func(values, timestamps);
    for rc in rcs.iter() {
        if let Some(tsm) = newTimeseriesMap(funcName, keep_metric_names, shared_timestamps, &ts_sq.metric_name) {
            let samples_scanned = rc.DoTimeseriesMap(tsm, values, timestamps);
            atomic.AddUint64(&samples_scanned_total, samples_scanned);
            tssLock.Lock();
            tsm.AppendTimeseriesTo(tss);
            tssLock.Unlock();
            continue
        }
        let ts: Timeseries = Timeseries::default();
        samplesScanned = do_rollup_for_timeseries(funcName, keep_metric_names, rc, &ts, &ts_sq.MetricName, values, timestamps, shared_timestamps)
        atomic.AddUint64(&samples_scanned_total, samplesScanned)
        tssLock.Lock();
        tss.push(&ts);
        tssLock.Unlock()
    }
    return (values, timestamps)
});
rowsScannedPerQuery.Update(float64(samples_scanned_total));
qt.Printf("rollup %s() over %d series returned by subquery: series=%d, samplesScanned=%d", funcName, len(tss_sq), tss.len(), samples_scanned_total)
return tss
}


fn get_keep_metric_names(expr: Expression) -> bool {
    match expr {
        Expression::AggrFuncExpr(ae) => {
            // Extract rollupFunc(...) from aggrFunc(rollupFunc(...)).
            // This case is possible when optimized aggrfn calculations are used
            // such as `sum(rate(...))`
            if ae.args.len() != 1 {
                return false
            }
            expr = ae.Args[0]
        }
        Expression::FuncExpr(fe) => {
            return fe.keep_metric_names;
        }
        _ => false
    }
}

fn do_parallel(tss: Vec<Timeseries>,
               f: fn(ts: &Timeseries, values: &[f64], timestamps: &[i64]) -> (&[f64], &[i64])) {
    let concurrency = cgroup.AvailableCPUs();
    if concurrency > tss.len() {
        concurrency = tss.len()
    }
    for i in 0 .. concurrency {
        let tmp_values: &[f64];
        let tmp_timestamps: &[i64];
        for ts in tss.iter() {
            (tmp_values, tmp_timestamps) = f(ts, tmp_values, tmp_timestamps)
        }
    }
}


fn eval_rollup_func_with_metric_expr(qt: &QueryTracer,
                                     ec: EvalConfig,
                                     func_name: String,
                                     rf: RollupFunc,
                                     expr: &Expression,
                                     me: &MetricExpr,
                                     iafc: IncrementalAggrFuncContext,
                                     window_expr: DurationExpr) -> RuntimeResult<Vec<Timeseries>> {
    let rollup_memory_size: i64;
    let window = window_expr.Duration(ec.step);
    if me.is_empty() {
        return eval_number(ec, f64::NAN)
    }

    // Search for partial results in cache.
    let (tssCached, start) = rollupResultCacheV.get(qt, ec, expr, window);
    if start > ec.End {
        // The result is fully cached.
        rollupResultCacheFullHits.Inc();
        return tssCached
    }
    if start > ec.Start {
        rollupResultCachePartialHits.Inc()
    } else {
        rollupResultCacheMiss.Inc()
    }

    // Obtain rollup configs before fetching data from db,
    // so type errors can be caught earlier.
    let shared_timestamps = getTimestamps(start, ec.end, ec.step, ec.MaxPointsPerSeries);
    let (pre_func, rcs, err) = getRollupConfigs(
        funcName, 
        rf, 
        expr, 
        start, 
        ec.End, 
        ec.Step, 
        ec.MaxPointsPerSeries, 
        window, 
        ec.LookbackDelta,
        shared_timestamps);

// Fetch the remaining part of the result.
    let tfs = searchutils.ToTagFilters(me.LabelFilters);
    let tfss = searchutils.JoinTagFilterss([][]storage.TagFilter{tfs}, ec.EnforcedTagFilterss)
    let mut min_timestamp = start - maxSilenceInterval;
    if window > ec.Step {
        min_timestamp -= window
    } else {
        min_timestamp -= ec.Step
    }
    let sq = storage.newSearchQuery(min_timestamp, ec.End, tfss, ec.MaxSeries)?;
    let rss = netstorage.ProcessSearchQuery(qt, sq, ec.Deadline);
    let rss_len = rss.len();
    if rss_len == 0 {
        rss.cancel();
        return mergeTimeseries(tssCached, nil, start, ec)?;
    }

    // Verify timeseries fit available memory after the rollup.
    // Take into account points from tssCached.
    let points_per_timeseries = 1 + (ec.End-ec.Start)/ec.Step;
    let mut timeseries_len = rss_len;
    if iafc != nil {
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
    let rollup_points = mulNoOverflow(points_per_timeseries, int64(timeseries_len *len(rcs)))
    let rollup_memory_size = mulNoOverflow(rollup_points, 16);
    let rml = get_rollup_memory_limiter();
if !rml.Get(uint64(rollup_memory_size)) {
rss.Cancel()
return nil, &UserReadableError{
Err: fmt.Errorf("not enough memory for processing %d data points across %d time series with %d points in each time series; "+
"total available memory for concurrent requests: %d bytes; "+
"requested memory: %d bytes; "+
"possible solutions are: reducing the number of matching time series; switching to node with more RAM; "+
"increasing -memory.allowedPercent; increasing `step` query arg (%gs)",
                rollup_points, timeseries_len *len(rcs), points_per_timeseries, rml.MaxSize, uint64(rollup_memory_size), float64(ec.Step)/1e3),
}
}
defer rml.Put(uint64(rollup_memory_size));

// Evaluate rollup
    let keep_metric_names = get_keep_metric_names(expr);
    let tss: Vec<Timeseries>;
    if iafc != nil {
        tss = eval_rollup_with_incremental_aggregate(
            qt,
            funcName,
            keep_metric_names,
            iafc,
            rss,
            rcs,
            pre_func,
            shared_timestamps)
    } else {
        tss = eval_rollup_no_incremental_aggregate(qt, funcName, keep_metric_names, rss, rcs, pre_func, shared_timestamps)
    }

    tss = merge_timeseries(tssCached, tss, start, ec);
    rollupResultCacheV.put(qt, ec, expr, window, tss);
    return tss
}


fn get_rollup_memory_limiter() -> MemoryLimiter {
return &rollupMemoryLimiter
}

fn eval_rollup_with_incremental_aggregate(qt: &QueryTracer,
                                          func_name: String,
                                          keep_metric_names: bool,
                                          iafc: &IncrementalAggrFuncContext,
                                          rss: &Results,
                                          rcs: &Vec<RollupConfig>,
                                          pre_func: PreFunc,
                                          shared_timestamps: &[i64]) -> RuntimeResult<Vec<Timeseries>> {
    let samples_scanned_total: u64;
    rss.runParallel(qt, |rs: Result, worker_id: usize| {
        (rs.values, rs.timestamps) = dropStaleNaNs(funcName, rs.values, rs.timestamps);
        pre_func(rs.values, rs.timestamps);
        let ts = getTimeseries();
        for rc in rcs.iter() {
            if let Some(tsm)= newTimeseriesMap(funcName, keep_metric_names, shared_timestamps, &rs.MetricName) {
                let samples_scanned = rc.doTimeseriesMap(tsm, rs.Values, rs.Timestamps);
                for ts in tsm.m {
                    iafc.updateTimeseries(ts, worker_id)
                }
                atomic.AddUint64(&samples_scanned_total, samples_scanned);
                continue
            }
            let samples_scanned = do_rollup_for_timeseries(funcName, keep_metric_names, rc, ts, &rs.MetricName, rs.Values, rs.Timestamps, shared_timestamps)
            atomic.AddUint64(&samples_scanned_total, samples_scanned);
            iafc.updateTimeseries(ts, worker_id);
        }
        return nil
    })?;
    let tss = iafc.finalizeTimeseries();
    rowsScannedPerQuery.Update(float64(samples_scanned_total))
    qt.Printf("series after aggregation with %s(): %d; samplesScanned=%d", iafc.ae.Name, tss.len(), samples_scanned_total)
    return tss
}

fn eval_rollup_no_incremental_aggregate(qt: &QueryTracer,
                                        func_name: String,
                                        keep_metric_names: bool,
                                        rss: Results,
                                        rcs: &Vec<RollupConfig>,
                                        pre_func: PreFunc,
                                        shared_timestamps: &[i64]) -> RuntimeResult<Vec<Timeseries>> {

    let tss: Vec<Timeseries> = Vec::with_capacity(rss.len() * rcs.len());
    let tssLock: Mutex;
    let mut samples_scanned_total: usize;
    rss.run_parallel(qt, |rs: Result, worker_id: usize| {
        (rs.values, rs.timestamps) = dropStaleNaNs(func_name, rs.values, rs.timestamps);
        pre_func(rs.values, rs.timestamps);
        for rc in rcs {
            if let Some(tsm) = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.MetricName) {
                let samples_scanned = rc.doTimeseriesMap(tsm, rs.values, rs.timestamps);
                atomic.AddUint64(&samples_scanned_total, samples_scanned);
                tssLock.Lock();
                tss = tsm.AppendTimeseriesTo(tss);
                tssLock.Unlock();
                continue
            }
            let ts: Timeseries;
            let mut samplesScanned = do_rollup_for_timeseries(funcName, keepMetricNames, rc, &ts, &rs.MetricName, rs.Values, rs.Timestamps, sharedTimestamps)
atomic.AddUint64(&samples_scanned_total, samplesScanned);
tssLock.Lock();
tss.push(&ts);
tssLock.Unlock()
}
return nil
})
rowsScannedPerQuery.Update(float64(samples_scanned_total))
qt.Printf("samplesScanned=%d", samples_scanned_total)
return tss
}

fn do_rollup_for_timeseries(func_name: String,
                            keep_metric_names: bool,
                            rc: RollupConfig,
                            ts_dst: Timeseries,
                            mn_src: &MetricName,
                            values_src: &[f64],
                            timestamps_src: &[i64],
                            shared_timestamps: &[i64]) -> u64 {
    ts_dst.metric_name.copy_from(mn_src);
    if len(rc.TagValue) > 0 {
        ts_dst.MetricName.AddTag("rollup", rc.TagValue)
    }
    if !keep_metric_names && !rollupFuncsKeepMetricName[func_name] {
        ts_dst.metric_name.reset_metric_group()
    }
    let samplesScanned: u64;
    (ts_dst.values, samplesScanned) = rc.do(ts_dst.values, values_src, timestamps_src);
    ts_dst.timestamps = shared_timestamps;
    return samplesScanned
}

var bbPool bytesutil.ByteBufferPool

fn mulNoOverflow(a: i64, b: i64) -> i64 {
if math.MaxInt64/b < a {
// Overflow
return math.MaxInt64
}
return a * b
}

fn dropStaleNaNs(funcName: String, values: &[f64], timestamps: &[i64]) -> (&[f64], &[i64]) {
    if noStaleMarkers || funcName == "default_rollup" || funcName == "stale_samples_over_time" {
    // Do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
    // since it uses them for Prometheus-style staleness detection.
    // Do not drop staleness marks for stale_samples_over_time() function, since it needs
    // to calculate the number of staleness markers.
        return (values, timestamps);
    }
// Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = false;
    for v in values {
        if decimal.IsStaleNaN(v) {
            has_stale_samples = true;
            break
        }
    }
    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return (values, timestamps);
    }
    // Slow path: drop Prometheus staleness marks from values.
    let mut k: usize = 0;
    let mut i: usize = 0;
    for i in 0 .. values.len() {
        let v = values[i];
        if decimal.IsStaleNaN(v) {
            continue
        }
        values[k] = v;
        timestamps[k] = timestamps[i];
        k = k + 1;
    }
    return dstValues, dstTimestamps
}