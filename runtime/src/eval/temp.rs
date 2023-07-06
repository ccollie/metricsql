use std::sync::Arc;
use std::task::Context;
use std::time::Duration;

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i32 = 50;

type Value = QueryValue;

/// QueryStats contains various stats for the query.
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
        let query = e.to_string();
        query = bytesutil.LimitStringLen(query, 300);
        let may_cache = ec.may_cache();
        trace_span!("eval: query={query}, timeRange={}, step={}, may_cache={may_cache}",
            ec.time_range_string(), ec.step)
    }
    let rv = eval_expr_internal(ctx, ec, e)?;
    if is_tracing {
        let series_count = rv.len();
        let mut points_per_series = 0;
        if rv.is_empty() {
            points_per_series = rv[0].len();
        }
        let points_count = series_count * points_per_series;
        trace!("series={series_count}, points={points_count}, points_per_series={points_per_series}")
    }
    return Ok(rv)
}

fn eval_expr_internal(ctx: &Arc<Context>, ec: &Arc<EvalConfig>, e: Expr) -> RuntimeResult<QueryValue> {
    match e {
        Expr::NumberExpr(n) => Ok(QueryValue::Scalar(n)),
        Expr::DurationExpr(de) => {
            let d = de.duration(ec.step);
            let d_sec = d / 1000_f64;
            Ok(QueryValue::Scalar(d_sec))
        },
        Expr::StringExpr(se) => {
            eval_string(ec, se)
        }
        Expr::MetricExpression(me) => {
            let re = RollupExpr::default_rollup(me);
            eval_rollup_func(ctx, ec, "default_rollup", rollup_default, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", me, err))?;
        },
        Expr::RollupExpr(re) => {
            eval_rollup_func(ctx, ec, "default_rollup", rollup_default, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", me, err))?;
        }
        Expr::FuncExpr(fe) => {
            let name = fe.name();
            let nrf = get_rollup_func(fe.name);
            if nrf == None {
                trace_span!("transform {}()", fe.Name);
                let rv = eval_transform_func(ctx, ec, fe)?;
                qtChild.Donef("series={}", rv.len());
                Ok(rv)
            };
            let (args, re) = eval_rollup_func_args(qt, ec, &fe)?;
            let rf = nrf(args)?;
            eval_rollup_func(ctx, ec, name, rf, e, re, None)
                .map_error(|err| format!("cannot evaluate {}: {}", fe, err))
        }
        Expr::AggrFuncExpr(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(ctx, ec, ae)?;
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

async fn binop_ex(be: BinaryExpr) -> RuntimeResult<QueryValue> {
    match (&be.left, &be.right) {
        (Value::Float(left), Value::Float(right)) => {
            let value = binaries::scalar_binary_operations(token, left, right)?;
            Ok(Value::Float(value))
        }
        (Value::Vector(left), Value::Vector(right)) => {
            binaries::vector_bin_op(expr, &left, &right)?
        }
        (Value::Vector(left), Value::Float(right)) => {
            binaries::vector_scalar_bin_op(expr, &left, right).await?
        }
        (Value::Float(left), Value::Vector(right)) => {
            binaries::vector_scalar_bin_op(expr, &right, left).await?
        }
    }
}

fn eval_transform_func(ctx: &Arc<Context>, ec: &EvalConfig, fe: &FuncExpr) -> RuntimeResult<Vec<Timeseries>> {
    let (tf, args) = match fe.function {
        BuiltInFunction::Transform(func) => {
            let handler = get_transform_func(func);
            let args = match fe.function {
                TransformFunction::Union() => {
                    eval_exprs_in_parallel(ctx, ec, &fe.args)?
                },
                _ => {
                    eval_exprs_sequentially(ctx, ec, &fe.args)?
                }
            };
            (handler, args)
        }
        _ => {
            Err(RuntimeError::InvalidFunction(format!("unknown func {}", fe.name())))
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
        let (fe, nrf) = try_get_arg_rollup_func_with_metric_expr(ae);
        if let Some(fe) = fe {
            // There is an optimized path for calculating AggrFuncExpr over rollupFunc over MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = eval_rollup_func_args(ctx, ec, fe)?;
            let rf = nrf(args);
            let iafc = newIncrementalAggrFuncContext(ae, callbacks);
            return eval_rollup_func(ctx, ec, fe.Name, rf, ae, re, iafc)
        }
    }
    let args = eval_exprs_in_parallel(ctx, ec, &ae.args)?;
    let af = getAggrFunc(ae.Name);
    let afa = AggrFuncArg{ ae, args, ec};

    trace!("eval {}", ae.name);

    af(afa)
        .map_error(|err| format!("cannot evaluate {}: {}", ae, err))
}

fn map_err_handler(e: Error) -> RuntimeError
    let msg = "cannot execute {}: {}", be, err)
}

fn eval_binary_op(ctx: &Arc<Context>, ec: &Arc<EvalConfig>, be: &BinaryOpExpr) -> RuntimeResult<Vec<Timeseries>> {
    let bf = get_binary_op_func(be.op);

    let (tss_left, tss_right) = match be.op {
        Operator::And | Operator::If => {
            // Fetch right-side series at first, since it usually contains
            // lower number of time series for `and` and `if` operator.
            // This should produce more specific label filters for the left side of the query.
            // This, in turn, should reduce the time to select series for the left side of the query.
            exec_binary_op_args(ctx, ec, &be.right, &be.left, be)?;
        }
        _ => {
            exec_binary_op_args(ctx, ec, be.left, be.right, be)?;
        }
    };

    let mut bfa = BinaryOpFuncArg {
        be: &be,
        left: &tss_left,
        right: &tss_right,
    };

    bf(bfa).map_err(|err| format!("cannot evaluate {be}: {}", err))
}


fn exec_binary_op_args(ctx: &Arc<Context>,
                       ec: &Arc<EvalConfig>,
                       expr_first: &Expr,
                       expr_second: &Expr,
                       be: &BinaryOpExpr) -> RuntimeResult((Vec<Timeseries>, Vec<Timeseries>)) {

    if !canPushdownCommonFilters(be) {
        // Execute expr_first and expr_second in parallel, since it is impossible to push-down common filters
        // from expr_first to expr_second.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886
        trace_span!("execute left and right sides of {} in parallel", be.op);

        trace!("expr1");
        let tss_first = eval_expr(ctx, ec, expr_first);

        trace!("expr2");
        let tss_second = eval_expr(ctx, ec, expr_second);

        wg.Wait();
        return Ok((tss_first, tss_second))
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
    if tss_first.is_empty() && be.op != Operator::Or {
        // Fast path: there is no sense in executing the expr_second when expr_first returns an empty result,
        // since the "expr_first op expr_second" would return an empty result in any case.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349
        return Ok((vec![], vec![]))
    }
    let mut lfs = get_common_label_filters(tss_first);
    let lfs = trim_filters_by_group_modifier(lfs, be);
    let expr_second = pushdown_binary_op_filters(expr_second, lfs);
    let tss_second = eval_expr(qt, ec, expr_second)?;

    Ok((tss_first, tss_second))
}


fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> RuntimeResult<(FuncExpr, newRollupFunc)> {
    if ae.args != 1 {
        return None, None
    }
    let e = ae.args[0];
// Make sure e contains one of the following:
// - metricExpr
// - metricExpr[d]
// - rollupFunc(metricExpr)
// - rollupFunc(metricExpr[d])

    match e {
        Expr::MetricExpr(me) => {
            if me.is_empty() {
return None, None
}
let fe = FuncExpr::default_rollup(me);
let nrf = getRollupFunc(fe.name);
return (fe, nrf)
}
Expr::RollupExpr(re) => {
    if me, ok: = re.Expr.(*MetricExpr);
    !ok | | me.is_empty() | | re.ForSubquery()
    {
        return None, None
    }
    // e = metricExpr[d]
    let fe = FuncExpr::default_rollup(re);
    let nrf = get_rollup_func(fe.name);
    return (fe, nrf)
}
}
if re, ok: = e.( *metricsql.RollupExpr); ok {
        if me,
        ok: = re.Expr.( * metricsql.MetricExpr); ! ok | | me.is_empty() | | re.ForSubquery() {
        return None, None
        }
// e = metricExpr[d]
        fe: = & metricsql.FuncExpr{
        Name: "default_rollup",
        Args: []metricsql.Expr{re,
    },
}
nrf: = getRollupFunc(fe.Name)
    return fe, nrf
    }
    fe, ok: = e.( * metricsql.FuncExpr)
    if ! ok {
    return None, None
    }
    let nrf = get_rollup_func(fe.name) ?;
    let rollupArgIdx = metricsql.GetRollupArgIdx(fe)
    if rollupArgIdx > = fe.args.len() {
    // Incorrect number of args for rollup func.
    return None, None
    }
    arg: = fe.args[rollupArgIdx]
    if me, ok: = arg.( * metricsql.MetricExpr); ok {
    if me.is_empty() {
    return None, None
    }
    // e = rollupFunc(metricExpr)
    return FuncExpr{
    Name: fe.Name,
    Args: []metricsql.Expr{me},
    }, nrf
    }
    if re, ok: = arg.( *metricsql.RollupExpr); ok {
    if me, ok: = re.Expr.( * metricsql.MetricExpr); ! ok || me.is_empty() | | re.ForSubquery() {
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

pub(super) fn eval_exprs_in_parallel(ec: &EvalConfig, es: &[Expr]) -> RuntimeResult<Vec<Vec<Timeseries>>> {
    if es.len() < 2 {
        return eval_exprs_sequentially(ec, es)
    }
    let rvs = Vec::with_capacity(es.len());
    trace!("eval function args in parallel");
    for e in es {
        trace!("eval arg {}", i);
        go func(e metricsql.Expr, i int) {
            let rv = eval_expr(ctx, ec, e)?;
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

    let rollup_arg_idx = fe.get_rollup_arg_idx();
    if fe.args.len() <= rollup_arg_idx {
        let msg = format!("expecting at least {} args to {}; got {} args; expr: {}",
                          rollup_arg_idx +1, fe.Name, fe.args.len, fe);
        return Err(RuntimeResult::General(msg))
    }

    let args = Vec::with_capacity( fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg);
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

// expr may contain:
// - rollupFunc(m) if iafc is None
// - aggrFunc(rollupFunc(m)) if iafc isn't None
fn eval_rollup_func(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func_name: &str,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: &IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {
    if re.at.is_none() {
        return eval_rollup_func_without_at(ctx, ec, func_name, rf, expr, re, iafc)
    }

    let at_timestamp = eval_at_expr(ctx, ec, re.at)
        .map_err(|err| UserReadableError::from_err(format!("cannot evaluate `@` modifier: {}", err)))?;

    let mut ec_new = ec.clone();
    ec_new.start = at_timestamp;
    ec_new.end = at_timestamp;
    let mut tss = eval_rollup_func_without_at(ctx, ec_new, func_name, rf, expr, re, iafc)?;

    // expand single-point tss to the original time range.
    let timestamps = ec.get_shared_timestamps();
    for ts in tss.iter_mut() {
        let v = ts.values[0];
        ts.timestamps = Arc::clone(timestamps);
        ts.values = vec![v; timestamps.len()]
    }
    Ok(tss)
}

pub fn eval_rollup_func_without_at(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: &IncrementalAggrFuncContext) -> RuntimeResult<Vec<Timeseries>> {
    let ec_new = ec;
    let mut offset: i64 = 0;
    if let Some(ofs) = re.offset {
        offset = ofs.value(ec.step);
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
    let rvs = match re.expr {
        Expr::MetricExpression(me) => {
            eval_rollup_func_with_metric_expr(ctx,ec_new, func, rf, expr, me, iafc, re.window)?;
        }
        _=> {
            if iafc.is_some() {
                logger.Panicf("BUG: iafc must be None for rollup {} over subquery {}", func, re)
            }
            eval_rollup_func_with_subquery(ctx, ec_new, funcName, rf, expr, re)?;
        }
    };

    if func == RollupFunction::AbsentOverTime {
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

    Ok(rvs)
}

#[inline]
fn get_duration_value(duration: &Option<DurationExpr>, step: i64) -> i64 {
    if let Some(ofs) = duration {
        ofs.value(step)
    } else {
        0
    }
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
                rvs[0].values[i] = f64::NAN;
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

    let ec_sq = ec.clone();
    ec_sq.start -= window + maxSilenceInterval + step;
    ec_sq.end += step;
    ec_sq.step = step;
    ec_sq.max_points_per_series = *maxPointsSubqueryPerTimeseries;

    validate_max_points_per_series(ec_sq.start, ec_sq.end, ec_sq.step, ec_sq.max_points_per_series)
        .map_err(|err| format!("cannot evaluate subquery: {}", err))?;

    // unconditionally align start and end args to step for subquery as Prometheus does.
    (ec_sq.start, ec_sq.end) = align_start_end(ec_sq.start, ec_sq.end, ec_sq.step);
    let tss_sq = eval_expr(ctx, ec_sq, re.expr)?;
    if tss_sq.is_empty() {
        return Ok(vec![])
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
    let series_by_worker_id = tsw.byWorkerID;
    do_parallel(tss_sq, |tsSQ: &Timeseries, values: &mut [f64], timestamps: &mut [i64], worker_id: u64| -> RuntimeResult<(&[f64], &[i64])> {
        removeNanValues(values, timestamps, tsSQ.values, tsSQ.Timestamps);
        preFunc(values, timestamps);
        for rc in rcs {
            if let Some(tsm) = newTimeseriesMap(func_name, keep_metric_names,
                                                sharedTimestamps: shared_timestamps, &tsSQ.metric_name) {
                let samples_scanned = rc.DoTimeseriesMap(tsm, values, timestamps);
                samples_scanned_total, samples_scanned)
                series_by_worker_id[worker_id].tss = tsm.AppendTimeseriesTo(series_by_worker_id[worker_id].tss)
                continue
            }
            let ts = Timeseries::default();
            samplesScanned = do_rollup_for_timeseries(func_name, keep_metric_names, rc, &ts, &tsSQ.metric_name, values, timestamps, shared_timestamps)
            atomic.AddUi64(&samples_scanned_total, samplesScanned)
            series_by_worker_id[worker_id].tss = append(series_by_worker_id[worker_id].tss, &ts)
        }
return values, timestamps
})
tss := make([]*timeseries, 0, len(tss_sq)*len(rcs))
for i := range seriesByWorkerID {
tss = append(tss,
        seriesByWorkerID: series_by_worker_id[i].tss...)
}
putTimeseriesByWorkerID(tsw)

    rowsScannedPerQuery.Update(float64(samples_scanned_total))
    qt.Printf("rollup {}() over {} series returned by subquery: series={}, samplesScanned={}",
              funcName, len(tss_sq), len(tss), samples_scanned_total)
    return tss, None
}

fn eval_rollup_func_with_metric_expr(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func_name: string,
    rf: RollupFunc,
    expr: &Expr,
    me: &MetricExpr,
    iafc: &IncrementalAggrFuncContext,
    window_expr: &DurationExpr) -> RuntimeResult<QueryValue> {
    let rollup_memory_size: i64;
    let window = window_expr.value(ec.step);
    if ctx.tracing_enabled {
        trace_span!("rollup {func_name}(): timeRange={}, step={}, window={window}",
                         ec.time_range_string(), ec.step);
        defer! {
            trace!("neededMemoryBytes={}", rollup_memory_size)
        }
    }

    if me.is_empty() {
        return Ok(QueryValue::Number(f64::NAN));
    }

    // Search for partial results in cache.
    let (tss_cached, start) = rollupResultCacheV.get(qt, ec, expr, window)?;
    if start > ec.end {
        // The result is fully cached.
        rollupResultCacheFullHits.Inc();
        return Ok(tss_cached)
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
    let mut rss = process_search_query(qt, sq, ec.Deadline)?;
    let rss_len = rss.len();
    if rss_len == 0 {
        rss.cancel();
        mergeTimeseries(tss_cached, None, start, ec);
    }
    ec.query_stats.addSeriesFetched(rss_len);

    // Verify timeseries fit available memory after the rollup.
    // Take into account points from tss_cached.
    let points_per_timeseries = 1 + (ec.end-ec.start)/ec.step;
    let mut timeseries_len = rss_len;
    if let Some(iafc) = iafc {
        // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
        timeseries_len = cgroup.AvailableCPUs();
        if iafc.ae.modifier.op != "" {
            if iafc.ae.limit > 0 {
                // There is an explicit limit on the number of output time series.
                timeseries_len *= iafc.ae.limit
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
    let mut rollup_points = mulNoOverflow(points_per_timeseries, timeseries_len * rcs.len());
    let rollup_memory_size = sumNoOverflow(mulNoOverflow(i64(rss_len), 1000), mulNoOverflow(rollup_points, 16))
    let max_memory = i64(logQueryMemoryUsage.N);
    if max_memory > 0 && rollup_memory_size > max_memory {
        requestURI = ec.GetRequestURI()
        logger.Warnf("remoteAddr={}, requestURI={}: the {} requires {} bytes of memory for processing; " +
                         "logging this query, since it exceeds the -search.logQueryMemoryUsage={}; " +
                         "the query selects {} time series and generates {} points across all the time series; try reducing the number of selected time series",
                     ec.QuotedRemoteAddr, requestURI, expr,
                     rollupMemorySize: rollup_memory_size, max_memory,
                     timeseriesLen: timeseries_len * len(rcs),
                     rollupPoints: rollup_points)
    }
    if max_memory: = i64(maxMemoryPerQuery.N);
    max_memory > 0 && rollup_memory_size > maxMemory {
        rss.cancel()
        return None,
        &UserReadableError{
        Err: fmt.Errorf("not enough memory for processing {}, which returns {} data points across {} time series with {} points in each time series " +
                            "according to -search.maxMemoryPerQuery={}; requested memory: {} bytes; " +
                            "possible solutions are: reducing the number of matching time series; increasing `step` query arg (step=%gs); " +
                            "increasing -search.maxMemoryPerQuery",
                        expr, rollup_points, timeseries_len * len(rcs), points_per_timeseries, max_memory, rollup_memory_size, float64(ec.step) / 1e3),
    }
}
    rml : = getRollupMemoryLimiter()
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
    trace_span!("rollup {func_name}() with incremental aggregation {}() over {} series; rollupConfigs={}",
        iafc.ae.Name, rss.Len(), rcs);

    let mut samples_scanned_total: u64;
    rss.run_parallel(qt, |rs: &Result, worker_id: usize| -> RuntimeResult<()> {
        (rs.values, rs.timestamps) = dropStaleNaNs(funcName, rs.values, rs.timestamps);
        pre_func(rs.values, rs.timestamps);

        let ts = getTimeseries();
        for rc in rcs {
            if let Some(tsm) = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.metric_name) {
                let samples_scanned = rc.do_timeseries_map(tsm, rs.values, rs.timestamps);
                for ts in tsm.m {
                    iafc.updateTimeseries(ts, worker_id)
                }
                atomic.AddUi64(&samplesScannedTotal, samples_scanned);
                continue
            }
            ts.reset();
            let samples_scanned = doRollupForTimeseries(
                funcName,
                keepMetricNames,
                rc,
                ts,
                &rs.metric_name,
                rs.values,
                rs.timestamps,
                sharedTimestamps);
            atomic.AddUi64(&samplesScannedTotal, samples_scanned);
            iafc.updateTimeseries(ts, worker_id)
        }
        return None
    });
let tss = iafc.finalizeTimeseries();
    rowsScannedPerQuery.Update(float64(samples_scanned_total))
    qt.Printf("series after aggregation with {}(): {}; samplesScanned={samples_scanned_total}",
              iafc.ae.name,
              tss.len());
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
    rss.run_parallel(qt, |rs: &Result, worker_id: usize| -> RuntimeResult<()>  {
        dropStaleNaNs(func_name, rs.values, rs.timestamps);
        pre_func(rs.values, rs.timestamps);
        for rc in rcs {
            if let Some(tsm) = newTimeseriesMap(funcName, keepMetricNames, sharedTimestamps, &rs.metric_name) {
                let samples_scanned = rc.doTimeseriesMap(tsm, rs.values, rs.timestamps);
                atomic.AddUi64(&samplesScannedTotal, samples_scanned);
                seriesByWorkerID[worker_id].tss = tsm.AppendTimeseriesTo(seriesByWorkerID[worker_id].tss)
                continue
            }
            let mut ts = Timeseries::default();
            let samples_scanned = doRollupForTimeseries(funcName, keepMetricNames, rc, &ts, &rs.metric_name, rs.Values, rs.Timestamps, sharedTimestamps)
            atomic.AddUi64(&samplesScannedTotal, samples_scanned);
            seriesByWorkerID[worker_id].tss = append(seriesByWorkerID[worker_id].tss, &ts)
        }
        return None
    });
    let mut tss = Vec::with_capacity(series_len * rcs.len());
    for i in 0..seriesByWorkerID {
        tss.push(series_by_worker_id[i].tss...)
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
    let (values, samples_scanned) = rc.do(&mut ts_dst, values_src, timestamps_src);
    ts_dst.timestamps = shared_timestamps;
    ts_dst.values = values;
    return samples_scanned
}