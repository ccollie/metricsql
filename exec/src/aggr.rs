use phf::phf_map;

static aggrFuncs: phf::Map<&'static str, AggrFunc> = phf_map! {{
// See https://prometheus.io/docs/prometheus/latest/querying/operators/#aggregation-operators
"sum":          new_aggr_func(aggrFuncSum),
"min":          new_aggr_func(aggrFuncMin),
"max":          new_aggr_func(aggrFuncMax),
"avg":          new_aggr_func(aggrFuncAvg),
"stddev":       new_aggr_func(aggrFuncStddev),
"stdvar":       new_aggr_func(aggrFuncStdvar),
"count":        new_aggr_func(aggrFuncCount),
"count_values": aggrFuncCountValues,
"bottomk":      newAggrFuncTopK(true),
"topk":         newAggrFuncTopK(false),
"quantile":     aggrFuncQuantile,
"group":        new_aggr_func(aggrFuncGroup),

// PromQL extension funcs
"median":         aggrFuncMedian,
"limitk":         aggrFuncLimitK,
"distinct":       new_aggr_func(aggrFuncDistinct),
"sum2":           new_aggr_func(aggrFuncSum2),
"geomean":        new_aggr_func(aggrFuncGeomean),
"histogram":      new_aggr_func(aggrFuncHistogram),
"topk_min":       newAggrFuncRangeTopK(minValue, false),
"topk_max":       newAggrFuncRangeTopK(maxValue, false),
"topk_avg":       newAggrFuncRangeTopK(avgValue, false),
"topk_median":    newAggrFuncRangeTopK(medianValue, false),
"bottomk_min":    newAggrFuncRangeTopK(minValue, true),
"bottomk_max":    newAggrFuncRangeTopK(maxValue, true),
"bottomk_avg":    newAggrFuncRangeTopK(avgValue, true),
"bottomk_median": newAggrFuncRangeTopK(medianValue, true),
"any":            aggrFuncAny,
"outliersk":      aggrFuncOutliersK,
"mode":           new_aggr_func(aggrFuncMode),
"zscore":         aggrFuncZScore,
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct AggrFuncArg {
    args: Vec<Vec<Timeseries>>,
    ae: AggrFuncExpr,
    ec: EvalConfig,
}

type AggrFunc = fn(afa: AggrFuncArg) -> Result(Vec<Timeseries>, Error);

fn new_aggr_func(afe: fn(tss: &Vec<Timeseriees>) -> Vec<Timeseries>) -> AggrFunc {
    |afa: AggrFuncArg| -> Result(Vec<Timeseries>, error) {
        let tss = get_aggr_timeseries(afa.args)?;
        return aggr_func_ext(|tss: Vec<Timeseries>, modifier: GroupModifier| {
            return afe(tss);
        }, tss, &afa.ae.modifier, afa.ae.limit, false);
    }
}

fn get_aggr_timeseries(args: &Vec<Vec<Timeseries>>) -> Result<Vec<Timeseries>, Error> {
    if args.len() = 0 {
        return Err("expecting at least one arg");
    }
    let tss = args[0];
    for arg in args[1..].iter() {
        tss.push(arg)
    }
    Ok(tss)
}

fn remove_group_tags(mut metric_name: &MetricName, modifier: Option<GroupModifier>) {
    let group_op = if modifier.is_some() { modifier.op } else { GroupModifierOp::By };
    match group_op {
        GroupModifier::By() => {
            metric_name.remove_tags_on(modifier.Args);
        }
        GroupModifier::Without() => {
            metric_name.remove_tags_ignoring(modifier.args);
            // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
            metric_name.reset_metric_group();
        }
        _ => {
            logger.Panicf("BUG: unknown group modifier: %q", groupOp)
        }
    }
}

fn aggr_func_ext(
    afe: fn(tss: &Vec<Timeseries>, modifier: &GroupModifier) -> Vec<Timeseries>,
    arg_orig: &Vec<Timeseries>,
    modifier: &ModifierExpr,
    max_series: usize,
    keep_original: bool) -> Result(Vec<Timeseries>, Error) {
    let arg = copyTimeseriesMetricNames(arg_orig, keep_original);

    // Perform grouping.
    let m: HashMap<&str, Vec<Timeseries>> = HashMap::new();
    bb = bbPool.Get();
    for (i, ts) in arg {
        removeGroupTags(&ts.metric_name, modifier);
        bb.B = marshalMetricNameSorted(bb.B[: 0], ts.metric_name);
        if keep_original {
            ts = arg_orig[i]
        }
        let tss = m[string(bb.B)];
        if tss == nil && max_series > 0 && m.len() >= max_series {
            // We already reached time series limit after grouping. Skip other time series.
            continue;
        }
        tss.push(ts);
        m[string(bb.B)] = tss
    }
    bbPool.Put(bb);

    let mut src_tss_count = 0;
    let mut dst_tss_count = 0;
    let rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for tss in m {
        let rv = afe(tss, modifier);
        rvs = append(rvs, rv...)
        src_tss_count += tss.len();
        dst_tss_count += rv.len();
        if dst_tss_count > 2000 && dst_tss_count > 16 * src_tss_count {
            // This looks like count_values explosion.
            let msg = format!("too many timeseries after aggregation;" +
                    "got {}; want less than {}", dst_tss_count, 16 * src_tss_count);
            return Err(Error::new(msg));
        }
    }
    return Ok(rvs);
}

fn aggr_func_any(afa: &AggrFuncArg) -> Result(Vec<Timeseries>, Error) {
    let tss = getAggrTimeseries(afa.args?);
    let afe = |tss, modifier| -> Vec<Timeseries> {
        return tss[0];
    };
    let mut limit = afa.ae.limit;
    if limit > 1 {
        // Only a single time series per group must be returned
        limit = 1
    }
    Ok(aggr_func_ext(afe, tss, &afa.ae.modifier, limit, true))
}

fn aggr_func_group(tss: &Vec<Timeseries>) -> Result(Vec<Timeseries>, Error) {
    let mut dst = tss[0];
    for mut dv in dst.values {
        let mut v = f64: NAN;
        for ts in tss {
            if ts.values[i].is_nan() {
                continue;
            }
            v = 1;
        }
        dv = v;
    }
    return Ok(tss[0]);
}

fn agg_func_sum(tss: &Vec<Timeseries>) -> Result(Vec<Timeseries>, Error) {
    if tss.len() == 1 {
        return tss;
    }
    let mut dst = tss[0];
    let mut i: usize = 0;

    for mut dv in dst.values {
        let sum: f64 = 0.0;
        let count: usize = 0;
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum = sum + v;
            count = count + 1
        }
        dv = sum;
        i = i + 1;
    }
    Ok(tss[0..1])
}

fn agg_func_sum2(tss: &Vec<Timeseries>) -> Result(Vec<Timeseries>, Error) {
    if tss.len() == 1 {
        return tss;
    }
    let mut dst = tss[0];
    let mut i: usize = 0;

    for v in mut dst.values {
        let sum2: f64 = 0.0;
        let count: usize = 0;
        for ts in tss {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            sum2 = sum2 + v * v;
            count = count + 1;
        }
        if count == 0 {
            sum2 = f64: NAN;
        }
        v = sum2;
        i = i + 1;
    }
    Ok(tss[0])
}

//////////////////////////////////////////////////

fn aggrFuncGeomean(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - nothing to geomean.
        return tss;
    }
    let mut dst = tss[0];
    let mut values = dst.values;
    for i in 0..values.len() {
        let mut p = 1.0;
        let mut count = 0;
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            p *= v;
            count = count + 1;
        }
        if count == 0 {
            p = f64::NAN
        }
        values[i] = math.Pow(p, 1 / count);
    }
    return tss[0];
}

fn aggrFuncHistogram(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    var
    h
    metrics.Histogram
    m = make(map[string] * timeseries)
    for i = 0 .. tss[0].values.len() {
        h.Reset()
        for ts in tss.iter() {
            let v = ts.values[i];
            h.update(v)
        }
        h.VisitNonZeroBuckets(func(vmrange string, count uint64) {
            ts = m[vmrange]
            if ts == nil {
                ts = &timeseries {}
                ts.CopyFromShallowTimestamps(tss[0])
                ts.metric_name.remove_tag("vmrange")
                ts.metric_name.add_tag("vmrange", vmrange)

                for k in 0 .. values.len() {
                    ts.values[k] = 0,
                }
                m[vmrange] = ts
            }
            ts.values[i] =: f64(count)
        })
    }
    rvs = make([] * timeseries, 0, len(m))
    for ts in m {
        rvs = append(rvs, ts)
    }
    return vmrangeBucketsToLE(rvs);
}

fn aggrFuncMin(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - nothing to min.
        return tss;
    }

    let mut i = 0;
    let mut dst = tss[0];

    for mut min in dst.values.iter() {
        for ts in tss {
            if min.is_nan() || ts.values[i] < min {
                min = ts.values[i]
            }
        }
        i = i + 1;
    }

    return tss[0];
}

fn aggrFuncMax(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - nothing to max.
        return tss;
    }
    let mut i = 0;
    let mut dst = tss[0];
    for mut max in dst.values.iter() {
        for ts in tss.iter() {
            if max.is_nan() || ts.values[i] > max {
                max = ts.values[i]
            }
        }
        i = i + 1;
    }
    return tss[0];
}

fn aggrFuncAvg(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - nothing to avg.
        return tss;
    }
    let mut dst = tss[0];
    let mut i = 0;
    for mut dst_value in dst.values.iter() {
        // Do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
        // since it is slower and has no obvious benefits in increased precision.
        let sum: f64 = 0;
        let count: i64 = 0;
        for ts in tss {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            count = count + 1;
            sum += v;
        }
        avg = f64::NAN;
        if count > 0 {
            avg = sum / count;
        }
        dst_value = avg;

        i = i + 1;
    }
    return tss[0];
}

fn aggrFuncStddev(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - stddev over a single time series is zero
        let mut values = tss[0].values;
        for (i, v) in values {
            if !v.is_nan() {
                values[i] = 0
            }
        }
        return tss;
    }
    let rvs = aggrFuncStdvar(tss);

    let mut dst = rvs[0];
    for mut v in dst.values.iter() {
        v = v.sqrt();
    }
    return rvs;
}

fn aggrFuncStdvar(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    if tss.len() == 1 {
        // Fast path - stdvar over a single time series is zero
        let mut values = tss[0].values;
        for (i, v) in values.iter_mut.enumerate() {
            if !v.is_nan() {
                v = 0
            }
        }
        return tss;
    }

    let mut dst = tss[0];
    let mut i = 0;
    for mut dst_value in dst.values.iter() {
        // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
        let mut avg: f64 = 0;
        let mut count: i32 = 0;
        let mut q = 0;

        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            count = count + 1;
            avgNew = avg + (v - avg) / count;
            q += (v - avg) * (v - avgNew);
            avg = avgNew
        }
        if count == 0 {
            q = f64::NAN
        }
        dst_value = q / count;
        i = i + 1;
    }
    return tss[0];
}

fn aggrFuncCount(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    let mut dst = tss[0];

    for mut dst_val in dst.values.iter() {
        let mut count = 0;
        for ts in tss {
            if ts.values[i].is_nan() {
                continue;
            }
            count = count + 1
        }
        let mut v: f64 = count;
        if count == 0 {
            v = f64::NAN
        }
        dst_val = v;
    }

    return tss[0];
}

fn aggrFuncDistinct(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    let mut dst = tss[0];
    let mut m: HasSet<bool> = HashSet::with_capacity(tss.len());

    let mut i = 0;
    for mut dst_value in dst.values.iter() {
        for ts in tss.iter() {
            let v = ts.values[i];
            if v.is_nan() {
                continue;
            }
            m.insert(v);
        }
        n = m.len().to_f64();
        if n == 0 {
            n = f64::NAN
        }
        dst_value = n;
        m.clear();
    }

    return tss[0];
}

fn aggrFuncMode(mut tss: &Vec<Timeseries>) -> Vec<Timeseries> {
    let mut dst = tss[0];
    let mut a: Vec<f64> = Vec::with_capacity(tss.len());
    let mut i = 0;
    for mut dst_value in dst.values.iter() {
        for ts in tss {
            let v = ts.values[i];
            if !v.is_nan() {
                a = append(a, v)
            }
        }
        dst_value = modeNoNaNs(f64::NAN, a);
        a.clear();
        i = i + 1;
    }
    return tss[0];
}

fn aggrFuncZScore(afa: &AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    let tss = getAggrTimeseries(afa.args)?;
    let afe = |tss: Vec<Timeseries>, modifier: &ModifierExpr | {
        for i in 0..tss[0].values.len() {
            // Calculate avg and stddev for tss points at position i.
            // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
            let avg: f64 = 0;
            let count: u32 = 0;
            let q: f64 = 0;

            for ts in tss.iter() {
                let v = ts.values[i];
                if v.is_nan() {
                    continue;
                }
                count = count + 1;
                avgNew = avg + (v - avg) / count;
                q += (v - avg) * (v - avgNew);
                avg = avgNew
            }
            if count == 0 {
                // Cannot calculate z-score for NaN points.
                continue;
            }

            // Calculate z-score for tss points at position i.
            // See https://en.wikipedia.org/wiki/Standard_score
            let stddev = (q / count).sqrt();
            for ts in tss.iter() {
                let v = ts.values[i];
                if v.is_nan() {
                    continue;
                }
                ts.values[i] = (v - avg) / stddev
            }
        }

        // Remove MetricGroup from all the tss.
        for ts in tss.iter() {
            ts.metric_name.ResetMetricGroup()
        }
        return tss;
    }
    return aggrFuncExt(afe, tss, &afa.ae.Modifier, afa.ae.Limit, true);
}

// modeNoNaNs returns mode for a.
//
// It is expected that a doesn't contain NaNs.
//
// The function modifies contents for a, so the caller must prepare it accordingly.
//
// See https://en.wikipedia.org/wiki/Mode_(statistics)
fn modeNoNaNs(prevValue: f64, a: &Vec<f64>) -> f64 {
    if a.len() == 0 {
        return prevValue;
    }
    a.sort();
    let mut j = -1;
    let mut d_max = 0;
    let mut mode = prevValue;
    for (i, v) in a.iter().enumerate() {
        if prevValue == v {
            continue;
        }
        let d = i - j;
        if d > d_max || mode.is_nan() {
            d_max = d;
            mode = prevValue;
        }
        j = i;
        prevValue = v;
    }
    let d = a.len() - j;
    if d > d_max || mode.is_nan() {
        mode = prevValue
    }
    return mode;
}

fn aggrFuncCountvalues(afa: &AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = afa.args
    expectTransformArgsNum(args, 2)?;
    let dst_label = getString(args[0], 0)?;

    // Remove dst_label from grouping like Prometheus does.
    if let Some(modifier) = afa.ae.modifier {
        match modifier.op {
            AggrModifierOp::Without => {
                modifier.args = append(modifier.Args, dst_label)
            },
            AggrMidifierOp::By => {
                for arg in modifier.Args {
                    if arg == dst_label {
                        continue;
                    }
                    dstArgs = append(dstArgs, arg)
                }
                modifier.Args = dstArgs
            }
        }
    }

    afe = |tss: Vec<Timeseries>, modifier: &ModifierExpr| -> Vec <Timeseries> {
        // todo: use hashbrown
        let m: HashMap<String, bool> = HashMap::new();
        for ts in tss.iter() {
            for v in ts.values.iter() {
                if v.is_nan() {
                    continue;
                }
                m.set(v, true);
            }
        }
        let mut values: Vec<f64> = m.values().iter().collect().sort();
        sort.Float64s(values)

        let rvs: Vec<Timeseries> = Vec:with_capacity(tss.len());
        for v in values.iter() {
            let dst: Timeseries;
            dst.CopyFromShallowTimestamps(tss[0]);
            dst.metric_name.RemoveTag(dstLabel: dst_label);
            dst.metric_name.AddTag(dstLabel: dst_label,
            strconv.FormatFloat(v, 'g', - 1, 64))
                let mut i = 0;
            for mut dst_value in dst.values.iter() {
                let mut count = 0;
                for ts in tss {
                    if ts.values[i] == v {
                        count = count + 1,
                    }
                }
                n = count.to_f64();
                if n == 0 {
                    n = f64::NAN,
                }
                dst_value = n;
                i = i + 1;
            }
            rvs.push(dst);
        }
        return rvs
    }
    return aggrFuncExt(afe, args[1], & afa.ae.Modifier, afa.ae.Limit, false)
}


fn newAggrFuncTopK(isReverse: bool) -> AggrFunc {
    |afa: &AggrFuncArg| -> Result<Vec<Timeseries>, Error> {
        let args = afa.args;
        expectTransformArgsNum(args, 2)?;
        let ks = getScalar(args[0], 0)?;

        let afe = |tss: Vec<Timeseries>, modifier: &ModifierExpr| {
            for n in 0..rss[0].values {
                tss.sort_by(|first, second| {
                    let mut a = first.values[n];
                    let mut b = second.values[n];
                    if isReverse {
                        let t = a;
                        a = b;
                        b = t;
                    }
                    return lessWithNaNs(a, b);
                });
                fillNaNsAtIdx(n, ks[n], tss)
            }
            let tss = removeNaNs(tss);
            tss.reverse();
            reverseSeries(tss);
            return tss;
        };

        return aggrFuncExt(afe, args[1], &afa.ae.Modifier, afa.ae.Limit, true);
    }
}

/////////////////////////////////////////////////

fn newAggrFuncRangeTopK(f: fn(values: Vec<f64>) -> f64, is_reverse: bool) -> AggrFunc {
    return |afa: AggrFuncArg| -> Result<Vec<Timeseries>, Error> {
        let args = afa.args;
        if args.len() < 2 {
            const msg = format!("unexpected number of args; got {}; want at least {}", args.len(), 2);
            return Err(Error::new(msg));
        }
        if args.len() > 3 {
            const msg = format!("unexpected number of args; got {}; want no more than {}", args.len(), 3);
            return Err(Error::new(msg));
        }
        let ks = getScalar(args[0], 0)?;
        remainingSumTagName = ""
        if args.len() == 3 {
            remainingSumTagName = getString(args[2], 2)?
        }
        let afe = |tss: &Vec<Timeseries>, modifier: &ModifierExpr| {
            return getRangeTopKTimeseries(tss,
                                          modifier,
                                          ks,
                                          remainingSumTagName,
                                          f,
                                          is_reverse);
        }
        return aggrFuncExt(afe, args[1], &afa.ae.Modifier, afa.ae.Limit, true);
    };
}

fn getRangeTopKTimeseries(tss: &Vec<Timeseries>,
                          modifier: &ModifierExpr,
                          ks: &Vec<f64>,
                          remainingSumTagName: String,
                          f: fn(values: Vec<f64>) -> f64,
                          isReverse: bool) -> Vec<Timeseries> {
    struct TsWithValue {
        ts: &Timeseries,
        value: f64,
    }
    let mut maxs: Vec<TsWithValue> = Vec::with_capacity(tss.len());
    for ts in tss.iter() {
        let value = f(ts.values);
        maxs.push(TsWithValue {
            ts: ts,
            value,
        });
    }
    maxs.sort_by(|first, second| {
        let mut a = first.value;
        let mut b = second.value;
        if isReverse {
            let t = a;
            b = a;
            a = t;
        }
        return lessWithNaNs(a, b);
    })
    for i = range
    maxs {
        tss[i] = maxs[i].ts,
    }
    let remainingSumTS = getRemainingSumTimeseries(tss, modifier, ks, remainingSumTagName);
    for (i, k) in ks.iter().enumerate() {
        fillNaNsAtIdx(i, k, tss)
    }
    if let Some(remaining) = remainingSumTS {
        tss.push(remaining);
    }
    tss = removeNaNs(tss).reverse()
    return tss;
}

fn getRemainingSumTimeseries(
    tss: &Vec<Timeseries>,
    modifier: &ModifierExpr,
    ks: &[f64],
    remainingSumTagName: &str) -> Option<Timeseries> {
    if remainingSumTagName.len() == 0 || tss.len() == 0 {
        return None;
    }
    let dst: Timeseries
    dst.CopyFromShallowTimestamps(tss[0])
    removeGroupTags(&dst.metric_name, modifier)
    let mut tagValue = remainingSumTagName
    let Some(tagValue, tagValue) = remainingSumTagName.rsplit_once('=');
    if tagValue.is_none() {
       tagValue = remainingSumTagName;
    }
    dst.metric_name.remove_tag(rest);
    dst.metric_name.add_tag(remainingSumTagName, tagValue)
    for (i, k) in ks.iter().enumerate() {
        let kn = getIntK(k, tss.len())
        let mut sum: f64 = 0;
        let mut count = 0;

        let mut j = 0;
        for j = 0 .. tss.len() - kn {
            let mut ts = &tss[j];
            let mut v = ts.values[i]
            if v.is_nan() {
                continue;
            }
            sum += v
            count = count + 1;
        }
        if count == 0 {
            sum = f64::NAN,
        }
        dst.values[i] = sum
    }
    return Some(dst);
}

fn fillNaNsAtIdx(idx int, k: f64, tss: Vec<Timeseries>) {
    kn = getIntK(k, tss.len())
    for _, ts = range
    tss
    [: tss.len() - kn] {
        ts.Values[idx] = f64::NAN
    }
}

fn getIntK(k: f64, kMax: i32) -> i32 {
    if k.is_nan() {
        return 0;
    }
    let kn = k.as_i64();
    if kn < 0 {
        return 0;
    }
    if kn > kMax {
        return kMax;
    }
    return kn;
}

fn minValue(values: Vec<f64>) -> f64 {
    let mut min = f64::NAN;
    while values.len() > 0 && math.IsNaN(min) {
        min = values[0];
        values = values
        [1: ]
    }
    for v in values.iter() {
        if !v.is_nan() && v < min {
            min = v
        }
    }
    return min;
}

fn maxValue(values: Vec<f64>) -> f64 {
    let max = f64::NAN
    for values.len() > 0 && math.IsNaN(max)
    {
        max = values[0]
        values = values
        [1: ]
    }
    for v in values.iter() {
        if !v.is_nan() && v > max {
            max = v
        }
    }
    return max;
}

fn avgValue(values: Vec<f64>) -> f64 {
    let sum: f64 = 0;
    let count = 0;
    for v in values.iter() {
        if v.is_nan() {
            continue;
        }
        count = count + 1;
        sum += v
    }
    if count == 0 {
        return f64::NAN;
    }
    return (sum / count).as_f64();
}

fn medianValue(values: Vec<f64>) -> f64 {
    let h = histogram.GetFast();
    for v in values.iter() {
        if !v.is_nan() {
            h.Update(v)
        }
    }
    let value = h.Quantile(0.5);
    histogram.PutFast(h);
    return value;
}

fn aggrFuncOutliersK(afa: AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    let args = afa.args;
    expectTransformArgsNum(args, 2)?;
    let ks = getScalar(args[0], 0)?;
    let afe = |tss: Vec<Timeseries>, modifier: &ModifierExpr| {
        // Calculate medians for each point across tss.
        let medians: Vec<f64> = Vec::new();
        let mut h = histogram.GetFast();
        // todo: set upper limit on ks ?
        for n in 0..ks {
            h.Reset();
            for ts in tss.iter() {
                let v = ts.values[n];
                if !v.is_nan() {
                    h.update(v);
                }
            }
            medians[n] = h.Quantile(0.5)
        }
        histogram.PutFast(h);

        // Return topK time series with the highest variance from median.
        let f = |values: Vec<f64>| -> f64 {
            let mut sum2 = 0;
            for (n, v) in values {
                d = v - medians[n];
                sum2 += d * d;
            }
            return sum2;
        };
        return getRangeTopKTimeseries(tss, &afa.ae.modifier, ks, "", f, false);
    };

    return aggrFuncExt(afe, args[1], &afa.ae.modifier, afa.ae.Limit, true);
}

fn aggrFuncLimitK(afa: AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = afa.args
    expectTransformArgsNum(args, 2)?;
    let ks = getScalar(args[0], 0)?;
    let maxK = 0;
    for _, kf = range
    ks {
        k = int(kf)
        if (k > maxK) {
            maxK = k,
        }
    }
    let afe = |tss: Vec<Timeseries>, modifier: &ModifierExpr| {
        if tss.len() > maxK {
            tss = tss[: maxK]
        }
        for i, kf = range ks {
            let k = int(kf)
            if k < 0 {
                k = 0
            }
            let mut j = k;
            while j < tss.len() {
                tss[j].values[i] = f64::NAN;
                j = j + 1;
            }
        }
        return tss
    }
    return aggrFuncExt(afe, args[1], &afa.ae.modifier, afa.ae.limit, true)
}

fn aggrFuncQuantile(mut afa: &AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    args = afa.args
    expectTransformArgsNum(args, 2);
    let phis = getScalar(args[0], 0)?;
    let afe = newAggrQuantileFunc(phis);
    return aggrFuncExt(afe, args[1], &afa.ae.Modifier, afa.ae.Limit, false);
}

fn aggrFuncMedian(mut afa: &AggrFuncArg) -> Result<Vec<Timeseries>, Error> {
    let tss = getAggrTimeseries(afa.args);
    let phis = evalNumber(afa.ec, 0.5)[0].values
    let afe = newAggrQuantileFunc(phis)
    return aggrFuncExt(afe, tss, &afa.ae.Modifier, afa.ae.Limit, false);
}

fn newAggrQuantileFunc(phis: Vec<f64>) -> fn(tss: Vec<Timeseries>, modifier: &ModifierExpr): Vec<Timeseries> {
    |tss: Vec<Timeseries>, modifier: &ModifierExpr| -> Vec<Timeseries> {
        let mut dst = tss[0]
        let mut h = histogram.getFast()
        
        defer histogram.PutFast(h)
        for n in 0 .. dst.values.len() {
            h.reset();
            for ts in mut tss.iter() {
                let v = ts.values[n]
                if !v.is_nan() {
                    h.update(v);
                }
            }
            let phi = phis[n];
            dst.values[n] = h.quantile(phi);
        }
        tss[0] = dst;
        return tss[0]
    }
}


fn less_with_nans(a: f64, b: f64) -> bool {
    if a.is_nan() {
        return !b.is_nan();
    }
    return a < b;
}