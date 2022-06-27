use phf::phf_map;
use crate::timeseries::Timeseries;

pub(crate) enum RollupArgValue {
    Int(i64),
    Float(f64),
    String(String),
    Timeseries(Timeseries),
}

const nan: f64 = f64::NAN;
const inf: f64 = f64::INFINITY;

// The maximum interval without previous rows.
const maxSilenceInterval: i64 = 5 * 60 * 1000;

pub type NewRollupFunc = fn(args: Vec<RollupArgValue>) -> Result<Rollupfunc, Error>;

// RollupFunc must return rollup value for the given rfa.
//
// prev_value may be nan, values and timestamps may be empty.
pub(super) type Rollupfunc = fn(rfa: RollupFuncArg) -> f64;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub(super) struct RollupFuncArg {
    // The value preceding values if it fits staleness interval.
    prev_value: f64,

    // The timestamp for prev_value.
    prev_timestamp: i64,

    // Values that fit window ending at curr_timestamp.
    values: Vec<f64>,

    // Timestamps for values.
    timestamps: Vec<i64>,

    // Real value preceding values without restrictions on staleness interval.
    real_prev_value: f64,

    // Real value which goes after values.
    real_next_value: f64,

    // Current timestamp for rollup evaluation.
    curr_timestamp: i64,

    // Index for the currently evaluated point relative to time range for query evaluation.
    idx: usize,

    // Time window for rollup calculations.
    window: i64,

    tsm: TimeseriesMap,
}

impl RollupFuncArg {
    pub fn reset(mut self) {
        self.prev_value = 0;
        self.prev_timestamp = 0;
        self.values = nil;
        self.timestamps = nil
        self.curr_timestamp = 0;
        self.idx = 0;
        self.window = 0;
        self.tsm = nil
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub(crate) struct RollupConfig {
    // This tag value must be added to "rollup" tag if non-empty.
    tagValue: String,
    fn: RollupFunc,
    start: i64,
    end: i64,
    step: i64,
    window: i64,

    // Whether window may be adjusted to 2 x interval between data points.
    // This is needed for functions which have dt in the denominator
    // such as rate, deriv, etc.
    // Without the adjustment their value would jump in unexpected directions
    // when using window smaller than 2 x scrape_interval.
    mayAdjustWindow: bool,

    timestamps: Vec<i64>,

    // LoookbackDelta is the analog to `-query.lookback-delta` from Prometheus world.
    lookbackDelta: i64,

    // Whether default_rollup is used.
    is_default_rollup: bool,
}

impl RollupConfig {
    // Do calculates rollups for the given timestamps and values, appends
    // them to dst_values and returns results.
    //
    // rc.Timestamps are used as timestamps for dst_values.
    //
    // timestamps must cover time range [rc.Start - rc.Window - maxSilenceInterval ... rc.End].
    //
    // Do cannot be called from concurrent goroutines.
    pub(crate) fn Do(&self, mut dst_values: &Vec<i64>, values: &Vec<i64>, timestamps: Vec<i64>) -> Vec<f64> {
        return rc.doInternal(dst_values, None, values, timestamps)
    }

    fn do_internal(&self, mut dst_values: &Vec<i64>, values: &Vec<i64>, timestamps: Vec<i64>) -> Vec<f64> {
        // Sanity checks.
        if self.step <= 0 {
            logger.Panicf("BUG: Step must be bigger than 0; got %d", rc.Step)
        }
        if self.start > rc.end {
            logger.Panicf("BUG: Start cannot exceed End; got %d vs %d", rc.Start, rc.End)
        }
        if self.window < 0 {
            logger.Panicf("BUG: Window must be non-negative; got %d", rc.Window)
        }
        if err := ValidateMaxPointsPerTimeseries(rc.Start, rc.End, rc.Step); err != nil {
            logger.Panicf("BUG: %s; this must be validated before the call to rollupConfig.Do", err)
        }

        // Extend dst_values in order to remove mallocs below.
        let dstValues = decimal.ExtendFloat64sCapacity(dst_values, rc.timestamps.len());

        let scrapeInterval = getScrapeInterval(timestamps);
        let mut max_prev_interval = getMaxPrevInterval(scrapeInterval);
        if rc.lookbackDelta > 0 && max_prev_interval > rc.lookbackDelta {
            max_prev_interval = rc.LookbackDelta
        }
        if minStalenessInterval > 0 {
            let msi = minStalenessInterval.Milliseconds();
            if msi > 0 && max_prev_interval < msi {
                max_prev_interval = msi
            }
        }
        let mut window = self.Window;
        if window <= 0 {
            window = self.step;
            if self.isDefaultRollup && rc.lookbackDelta > 0 && window > rc.lookbackDelta {
                // Implicit window exceeds -search.maxStalenessInterval, so limit it to -search.maxStalenessInterval
                // according to https://github.com/VictoriaMetrics/VictoriaMetrics/issues/784
                window = rc.lookbackDelta
            }
        }
        if self.mayAdjustWindow && window < max_prev_interval {
            window = max_prev_interval
        }
        let mut rfa = getRollupFuncArg();
        rfa.idx = 0;
        rfa.window = window;
        rfa.tsm = tsm;

        let mut i = 0;
        let mut j = 0;
        let mut ni = 0;
        let mut nj = 0;
        let mut f = rc.Func;
        for tEnd in self.timestamps.iter() {
            let mut tStart = tEnd - window;
            let ni = seekFirstTimestampIdxAfter(timestamps[i..], tStart, ni);
            i = i + ni;
            if j < i {
                j = i
            }
            let nj = seekFirstTimestampIdxAfter(timestamps[j..], tEnd, nj);
            j += nj;

            rfa.prevValue = nan;
            rfa.prevTimestamp = tStart - max_prev_interval;
            if i < timestamps.len() && i > 0 && timestamps[i-1] > rfa.prevTimestamp {
                rfa.prevValue = values[i-1];
                rfa.prevTimestamp = timestamps[i-1]
            }
            rfa.values = values[i..j];
            rfa.timestamps = timestamps[i..j];
            if i > 0 {
                rfa.realPrevValue = values[i-1]
            } else {
                rfa.realPrevValue = nan
            }
            if j < values.len() {
                rfa.realNextValue = values[j]
            } else {
                rfa.realNextValue = nan
            }
            rfa.currTimestamp = tEnd;
            let value = f(rfa);
            rfa.idx = rfa.idx + 1;
            dstValues = append(dstValues, value)
        }
        putRollupFuncArg(rfa);

        return dstValues
    }

    // DoTimeseriesMap calculates rollups for the given timestamps and values and puts them to tsm.
    pub(crate) fn doTimeseriesMap(&self, tsm: &TimeseriesMap, values: &Vec<f64>, timestamps: Vec<i64>) {
        let ts = getTimeseries();
        rc.doInternal(&mut ts.values, tsm, values, timestamps);
        putTimeseries(ts)
    }
}


fn newRollupFuncOneArg(rf: Rollupfunc) -> NewRollupFunc {
    |args: Vec<RollupArgValue>| -> Result<Rollupfunc, error> {
        expectRollupArgsNum(args, 1);
        return rf;
    }
}

fn newRollupFuncTwoArgs(rf: Rollupfunc) -> NewRollupFunc {
    |args: Vec<RollupArgValue>| -> Result<Rollupfunc, error> {
        expectRollupArgsNum(args, 2);
        return rf;
    }
}

fn seekFirstTimestampIdxAfter(timestamps: &[i64], seek_timestamp: i64, n_hint: usize) -> usize {
    let mut ts = timestamps;
    if timestamps.len() == 0 || timestamps[0] > seek_timestamp {
        return 0;
    }
    let mut start_idx = n_hint - 2;
    if start_idx < 0 {
        start_idx = 0
    }
    if start_idx >= timestamps.len() {
        start_idx = timestamps.len() - 1
    }
    let mut end_idx = n_hint + 2;
    if end_idx > timestamps.len() {
        end_idx = timestamps.len()
    }
    if start_idx > 0 && timestamps[start_idx] <= seek_timestamp {
        ts = &timestamps[start_idx..];
        end_idx -= start_idx
    } else {
        start_idx = 0
    }
    if end_idx < timestamps.len() && timestamps[end_idx] > seek_timestamp {
        ts = &timestamps[0..end_idx];
    }
    if timestamps.len() < 16 {
        // Fast path: the number of timestamps to search is small, so scan them all.
        for (i, timestamp) in ts.iter().enumerate() {
            if *timestamp > seek_timestamp {
                return start_idx + i;
            }
        }
        return start_idx + ts.len();
    }
    // Slow path: too big timestamps.len(), so use binary search.
    let i = binarySearchInt64(ts, seek_timestamp + 1);
    return start_idx + i;
}

fn binarySearchInt64(a: &[i64], v: i64) -> usize {
    // Copy-pasted sort.Search from https://golang.org/src/sort/search.go?s=2246:2286#L49
    let mut i: usize = 0;
    let mut j:  usize = a.len();

    while i < j {
        let h = (i + j) >> 1;
        if h < a.len() && a[h] < v {
            i = h + 1;
        } else {
            j = h
        }
    }
    return i
}


fn getScrapeInterval(timestamps: &[i64]) -> i64 {
    if timestamps.len() < 2 {
        return maxSilenceInterval;
    }

    // Estimate scrape interval as 0.6 quantile for the first 20 intervals.
    let mut ts_prev = timestamps[0];
    let mut timestamps = timestamps[1..];
    if timestamps.len() > 20 {
        timestamps = timestamps[0..20]
    }
    a = getFloat64s();
    let intervals = a.A[0..0];
    for ts in timestamps {
        intervals.push(ts - ts_prev);
        ts_prev = ts
    }
    let scrape_interval = int64(quantile(0.6, intervals))
    a.A = intervals
    putFloat64s(a)
    if scrape_interval <= 0 {
        return int64(maxSilenceInterval);
    }
    return scrape_interval;
}

fn getMaxPrevInterval(scrapeInterval: i64) -> i64 {
    // Increase scrapeInterval more for smaller scrape intervals in order to hide possible gaps
    // when high jitter is present.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/139 .
    if scrapeInterval <= 2 * 1000 {
        return scrapeInterval + 4 * scrapeInterval;
    }
    if scrapeInterval <= 4 * 1000 {
        return scrapeInterval + 2 * scrapeInterval;
    }
    if scrapeInterval <= 8 * 1000 {
        return scrapeInterval + scrapeInterval;
    }
    if scrapeInterval <= 16 * 1000 {
        return scrapeInterval + scrapeInterval / 2;
    }
    if scrapeInterval <= 32 * 1000 {
        return scrapeInterval + scrapeInterval / 4;
    }
    return scrapeInterval + scrapeInterval / 8;
}

fn removeCounterResets(values: &mut Vec<f64>) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from vmstorage.
    if values.len() == 0 {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    let mut i = 0;
    for v in values.iter_mut() {
        let d = v - prev_value;
        if d < 0.0 {
            if (-d * 8) < prev_value {
                // This is likely jitter from `Prometheus HA pairs`.
                // Just substitute v with prev_value.
                v = prev_value
            } else {
                correction += prev_value
            }
        }
        prev_value = v;
        values[i] = v + correction;
        i = i + 1;
    }
}

fn deltaValues(values: &mut [f64]) {
// There is no need in handling NaNs here, since they are impossible
// on values from vmstorage.
    if values.len() == 0 {
        return;
    }
    let mut prev_delta: f64 = 0.0;
    let mut prev_value = values[0];
    for (i, v) in values[1..] {
        prev_delta = v - prev_value;
        values[i] = prev_delta;
        prev_value = v
    }
    values[values.len() - 1] = prev_delta
}

fn derivValues(values: &mut [f64], timestamps: &Vec<i64>) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from vmstorage.
    if values.len() == 0 {
        return;
    }
    let mut prev_deriv: f64 = 0.0;
    let mut prev_value = values[0];
    let mut prev_ts = timestamps[0];
    for (i, v) in values[1..].iter().enumerate() {
        let ts = timestamps[i + 1];
        if ts == prev_ts {
            // Use the previous value for duplicate timestamps.
            values[i] = prev_deriv;
            continue;
        }
        dt = (ts - prev_ts) / 1e3;
        prev_deriv = (v - prev_value) / dt;
        values[i] = prev_deriv;
        prev_value = *v;
        prev_ts = ts
    }
    values[values.len() - 1] = prev_deriv
}

fn newRollupHoltWinters(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 3)?;
    let sfs = getScalar(args[1], 1)?;
    let tfs = getScalar(args[2], 2)?;

    move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        if values.len() == 0 {
            return rfa.prev_value;
        }
        sf = sfs[rfa.idx];
        if sf <= 0 || sf >= 1 {
            return nan;
        }
        tf = tfs[rfa.idx];
        if tf <= 0 || tf >= 1 {
            return nan;
        }

        // See https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing .
        // TODO: determine whether this shit really works.
        s0 = rfa.prev_value;
        if math.IsNaN(s0) {
            s0 = values[0];
            values = values[1..];
            if values.len() == 0 {
                return s0;
            }
        }
        b0 = values[0] - s0;
        for v in values {
            s1 = sf * v + (1 - sf) * (s0 + b0);
            b1 = tf * (s1 - s0) + (1 - tf) * b0;
            s0 = s1;
            b0 = b1
        }
        return s0;
    }
}

fn newRollupPredictLinear(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let secs = getScalar(args[1], 1)?;

    let f = |rfa: RollupFuncArg| -> f64 {
        let (v, k) = linearRegression(rfa);
        if v.is_nan() {
            return nan;
        }
        sec = secs[rfa.idx];
        return v + k * sec;
    };
    
    Ok(f)
}

fn linearRegression(rfa: RollupFuncArg) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut timestamps = rfa.timestamps;
    let n = values.len();
    if n == 0 {
        return (nan, nan);
    }
    if areConstValues(values) {
        return (values[0], 0.0);
    }

    // See https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    let intercept_time = rfa.curr_timestamp;
    let mut vSum: f64 = 0.0;
    let mut tSum: f64 = 0.0;
    let mut tvSum: f64 = 0.0;
    let mut ttSum: f64 = 0.0;
    for (i, v) in values {
        let dt = f64(timestamps[i] - intercept_time) / 1e3;
        vSum += v;
        tSum += dt;
        tvSum += dt * v;
        ttSum += dt * dt
    }
    let mut k: f64 = 0.0;
    let t_diff = ttSum - tSum * tSum / n;
    if t_diff.abs() >= 1e-6 {
        // Prevent from incorrect division for too small t_diff values.
        k = (tvSum - tSum * vSum / n) / t_diff;
    }
    v = vSum / n - k * tSum / n;
    return (v, k);
}

fn areConstValues(values: &Vec<f64>) -> bool {
    if values.len() <= 1 {
        return true;
    }
    vPrev = values[0];
    for v in values[1..] {
        if v != vPrev {
            return false;
        }
        vPrev = v
    }
    return true;
}

fn newRollupDurationOverTime(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let dMaxs = getScalar(args[1], 1)?;

   let f = move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut timestamps = rfa.timestamps;
        if timestamps.len() == 0 {
            return nan;
        }
        let tPrev = timestamps[0];
        let dSum: i64 = 0;
        let dMax = (dMaxs[rfa.idx] * 1000) as i64;
        for t in timestamps {
            let d = t - tPrev;
            if d <= dMax {
                dSum += d;
            }
            tPrev = t
        }
        return f64(dSum) / 1000
    };
    
    Ok(f)
}

fn newRollupShareLE(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupShareFilter(args, countFilterLE);
}

fn countFilterLE(values: &Vec<f64>, le: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if v <= le {
            n = n + 1;
        }
    }
    return n;
}

fn newRollupShareGT(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupShareFilter(args, countFilterGT);
}

fn countFilterGT(values: &[f64], gt: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if *v > gt {
            n = n + 1;
        }
    }
    return n;
}

fn countFilterEQ(values: &[f64], eq: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if v == eq {
            n = n + 1;
        }
    }
    return n;
}

fn countFilterNE(values: &[f64], ne: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if v != ne {
            n = n + 1;
        }
    }
    return n;
}

fn newRollupShareFilter(args: Vec<RollupArgValue>, countFilter: fn(values: &[f64], limit: f64) -> i32) -> Result<RollupFunc, Error> {
    let rf = newRollupCountFilter(args, countFilter)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let n = rf(rfa);
        return n / rfa.values.len();
    };
    Ok(f)
}

fn newRollupCountLE(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupCountFilter(args, countFilterLE);
}

fn newRollupCountGT(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupCountFilter(args, countFilterGT);
}

fn newRollupCountEQ(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupCountFilter(args, countFilterEQ);
}

fn newRollupCountNE(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return newRollupCountFilter(args, countFilterNE);
}

fn newRollupCountFilter(args: Vec<RollupArgValue>, countFilter: fn(values: &[f64], limit: f64) -> i32) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let limits = getScalar(args[1], 1)?;
    move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        if values.len() == 0 {
            return nan;
        }
        let limit = limits[rfa.idx];
        return countFilter(values, limit) as f64;
    }
}

fn newRollupHoeffdingBoundLower(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let phis = getScalar(args[0], 0)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let (bound, avg) = rollupHoeffdingBoundInternal(rfa, phis);
        return avg - bound;
    };

    Ok(f)
}

fn newRollupHoeffdingBoundUpper(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let phis = getScalar(args[0], 0)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let (bound, avg) = rollupHoeffdingBoundInternal(rfa, phis);
        return avg + bound;
    };

    Ok(f)
}

fn rollupHoeffdingBoundInternal(rfa: RollupFuncArg, phis: &[f64]) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return (nan, nan);
    }
    if values.len() == 1 {
        return (0, values[0]);
    }
    let vMax = rollupMax(rfa);
    let vMin = rollupMin(rfa);
    let vAvg = rollupAvg(rfa);
    let vRange = vMax - vMin;
    if vRange <= 0 {
        return (0, vAvg);
    }
    let phi = phis[rfa.idx];
    if phi >= 1 {
        return (inf, vAvg);
    }
    if phi <= 0 {
        return (0, vAvg);
    }
// See https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
// and https://www.youtube.com/watch?v=6UwcqiNsZ8U&feature=youtu.be&t=1237
    let bound = vRange * math.Sqrt(math.Log(1 / (1 - phi)) / (2 * float64(values.len())))
    return (bound, vAvg)
}

fn newRollupQuantiles(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    if args.len() < 3 {
        return nil;, fmt.Errorf("unexpected number of args: %d; want at least 3 args", args.len())
    }
    tssPhi, ok = args[0].([] * timeseries)
    if !ok {
        return nil;, fmt.Errorf("unexpected type for phi arg: %T; want string", args[0])
    }
    let phiLabel = getString(tssPhi, 0)?;
    let phiArgs = args[1 .. args.len() - 1];
    let phis = Vec::with_capacity(phiArgs.len());
    let phiStrs: Vec<String> = Vec::with_capacity(phiArgs.len());

    for (i, phiArg) in phiArgs {
        let phi_values = getScalar(phiArg, i + 1);
        if !Some(phi_values) {
            fmt.Errorf("cannot obtain phi from arg #%d: %w", i + 1, err)
        }
        phis[i] = phi_values[0];
        phiStrs[i] = format!("{}", phi_values[0]);
    }
    let f = move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        if values.len() == 0 {
            return rfa.prev_value;
        }
        if values.len() == 1 {
            // Fast path - only a single value.
            return values[0];
        }
        let qs = getFloat64s();
        qs.A = quantiles(qs.A[0..0], phis, values)
        let idx = rfa.idx;
        let tsm = rfa.tsm;
        for (i, phiStr) in phiStrs {
            let ts = tsm.GetOrCreateTimeseries(phiLabel, phiStr)
            ts.values[idx] = qs.A[i]
        }
        putFloat64s(qs);
        return nan;
    }

    Ok(f)
}

fn newRollupQuantile(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expectRollupArgsNum(args, 2)?;
    let phis = getScalar(args[0], 0)?;
    |rfa: RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        let phi = phis[rfa.idx];
        quantile(phi, values);
    };
    return rf;
}

fn rollupHistogram(rfa: RollupFuncArg) -> f64 {
    let mut values = rfa.values;
    tsm = rfa.tsm;
    tsm.h.Reset();
    for v in values {
        tsm.h.Update(v)
    }
    idx = rfa.idx;
    tsm.h.VisitNonZeroBuckets( fn (vmrange
    string, count
    uint64) {
        ts = tsm.GetOrCreateTimeseries("vmrange", vmrange)
        ts.Values[idx] =: f64(count)
    })
    return nan;
}

fn rollupAvg(rfa: RollupFuncArg) -> f64 {
// Do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
// since it is slower and has no significant benefits in precision.

// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
// Do not take into account rfa.prev_value, since it may lead
// to inconsistent results comparing to Prometheus on broken time series
// with irregular data points.
        return nan;
    }
    let sum: f64 = 0;
    for v in values {
        sum += v
    }
    return sum / values.len();
}

fn rollupMin(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
// Do not take into account rfa.prev_value, since it may lead
// to inconsistent results comparing to Prometheus on broken time series
// with irregular data points.
        return nan;
    }
    minValue = values[0];
    for v in values {
        if v < minValue {
            minValue = v
        }
    }
    return minValue;
}

fn rollupMax(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
// Do not take into account rfa.prev_value, since it may lead
// to inconsistent results comparing to Prometheus on broken time series
// with irregular data points.
        return nan;
    }
    maxValue = values[0];
    for v in values {
        if v > maxValue {
            maxValue = v
        }
    }
    return maxValue;
}

fn rollupTmin(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    let mut timestamps = rfa.timestamps;
    if values.len() == 0 {
        return nan;
    }
    minValue = values[0];
    minTimestamp = timestamps[0];
    for (i, v) in values {
        // Get the last timestamp for the minimum value as most users expect.
        if v <= minValue {
            minValue = v;
            minTimestamp = timestamps[i];
        }
    }
    return (minTimestamp / 1e3) as f64;
}

fn rollupTmax(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    let mut timestamps = rfa.timestamps;
    if values.len() == 0 {
        return nan;
    }
    let mut maxValue = values[0];
    let mut maxTimestamp = timestamps[0];
    for (i, v) in values {
        // Get the last timestamp for the maximum value as most users expect.
        if v >= maxValue {
            maxValue = v;
            maxTimestamp = timestamps[i];
        }
    }
    return (maxTimestamp / 1e3) as f64;
}
///////////////////////

fn rollupTfirst(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = rfa.timestamps;
    if timestamps.len() == 0 {
        // Do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return timestamps[0] / 1e3;
}

fn rollupTlast(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = rfa.timestamps;
    if timestamps.len() == 0 {
        // Do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return (timestamps[timestamps.len() - 1]) / 1e3;
}

fn rollupTlastChange(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    let mut timestamps = rfa.timestamps;
    let last = values.len() - 1;
    let last_value = values[last];
    values = &values[0..last];
    let mut i = last;
    for value in values.iter().rev() {
        if value != last_value {
            return timestamps[i + 1] / 1e3;
        }
        i = i - 1;
    }
    if rfa.prev_value.is_nan() || rfa.prev_value != last_value {
        return f64(timestamps[0]) / 1e3;
    }
    return nan;
}

fn rollupSum(rfa: RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
// Do not take into account rfa.prev_value, since it may lead
// to inconsistent results comparing to Prometheus on broken time series
// with irregular data points.
        return nan;
    }
    let mut sum: f64 = 0;
    for v in values {
        sum += v
    }
    return sum;
}

fn rollupRateOverSum(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change since rfa.prev_value.
        return 0;
    }
    dt = rfa.window;
    if !rfa.prev_value.is_nan() {
        dt = timestamps[timestamps.len() - 1] - rfa.prev_timestamp
    }
    sum = f64(0)
    for v in rfa.values {
        sum += v
    }
    return sum / (f64(dt) / 1e3);
}

fn rollupRange(rfa: RollupFuncArg) -> f64 {
    let max = rollupMax(rfa);
    let min = rollupMin(rfa);
    return max - min;
}

fn rollupSum2(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return rfa.prev_value * rfa.prev_value;
    }
    let mut sum2: f64 = 0;
    for v in values {
        sum2 += v * v
    }
    return sum2;
}

fn rollupGeomean(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return rfa.prev_value;
    }
    let mut p = 1.0;
    for v in values {
        p *= v
    }
    return math.Pow(p, 1 / f64(values.len()));
}

fn rollupAbsent(rfa: RollupFuncArg) -> f64 {
    if rfs.values.len() == 0 {
        return 1;
    }
    return nan;
}

fn rollupPresent(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfs.values.len() > 0 {
        return 1;
    }
    return nan;
}

fn rollupCount(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    return f64(values.len());
}

fn rollupStaleSamples(rfa: RollupFuncArg) -> f64 {
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    let mut n = 0;
    for v in rfa.values {
        if decimal.IsStaleNaN(v) {
            n = n + 1;
        }
    }
    return f64(n);
}

fn rollupStddev(rfa: RollupFuncArg) -> f64 {
    let stdVar = rollupStdvar(rfa);
    return math.Sqrt(stdVar);
}

fn rollupStdvar(rfa: RollupFuncArg) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    if values.len() == 1 {
        // Fast path.
        return 0;
    }
    let mut avg: f64 = 0;
    let mut count: f64 = 0;
    let mut q: f64 = 0;
    for v in values {
        count = count + 1;
        avgNew = avg + (v - avg) / count;
        q += (v - avg) * (v - avgNew);
        avg = avgNew
    }
    return q / count;
}

fn rollupIncreasePure(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    // restore to the real value because of potential staleness reset
    prevValue = rfa.real_prev_value;
    if math.IsNaN(prevValue) {
        if values.len() == 0 {
            return nan;
        }
        // Assume the counter starts from 0.
        prevValue = 0
    }
    if values.len() == 0 {
        // Assume the counter didn't change since prev_value.
        return 0;
    }
    return values[values.len() - 1] - prevValue;
}

fn rollupDelta(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    prevValue = rfa.prev_value;
    if math.IsNaN(prevValue) {
        if values.len() == 0 {
            return nan;
        }
        if !math.IsNaN(rfa.real_prev_value) {
            // Assume that the value didn't change during the current gap.
            // This should fix high delta() and increase() values at the end of gaps.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/894
            return values[values.len() - 1] - rfa.real_prev_value;
        }
        // Assume that the previous non-existing value was 0 only in the following cases:
        //
        // - If the delta with the next value equals to 0.
        //   This is the case for slow-changing counter - see https://github.com/VictoriaMetrics/VictoriaMetrics/issues/962
        // - If the first value doesn't exceed too much the delta with the next value.
        //
        // This should prevent from improper increase() results for os-level counters
        // such as cpu time or bytes sent over the network interface.
        // These counters may start long ago before the first value appears in the db.
        //
        // This also should prevent from improper increase() results when a part of label values are changed
        // without counter reset.
        let mut d: f64;
        if values.len() > 1 {
            d = values[1] - values[0]
        } else if !math.IsNaN(rfa.real_next_value) {
            d = rfa.real_next_value - values[0]
        }
        if d == 0 {
            d = 10
        }
        if math.Abs(values[0]) < 10 * (math.Abs(d) + 1) {
            prevValue = 0
        } else {
            prevValue = values[0];
            values = values[1..]
        }
    }
    if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0;
    }
    return values[values.len() - 1] - prevValue;
}

fn rollupDeltaPrometheus(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    // Just return the difference between the last and the first sample like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if values.len() < 2 {
        return nan;
    }
    return values[values.len() - 1] - values[0];
}

fn rollupIdelta(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change on the given interval.
        return 0;
    }
    let last_value = values[values.len() - 1];
    let values = values[0..values.len() - 1];
    if values.len() == 0 {
        prevValue = rfa.prev_value;
        if prevValue.is_nan() {
            // Assume that the previous non-existing value was 0.
            return last_value;
        }
        return last_value - prevValue;
    }
    return last_value - values[values.len() - 1];
}

fn rollupDerivSlow(rfa: RollupFuncArg) -> f64 {
    // Use linear regression like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/73
    let (_, k) = linearRegression(rfa);
    return k;
}

fn rollupDerivFast(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    timestamps = rfa.timestamps;
    prevValue = rfa.prev_value;
    prevTimestamp = rfa.prev_timestamp;
    if prevValue.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        if values.len() == 1 {
            // It is impossible to determine the duration during which the value changed
            // from 0 to the current value.
            // The following attempts didn't work well:
            // - using scrape interval as the duration. It fails on Prometheus restarts when it
            //   skips scraping for the counter. This results in too high rate() value for the first point
            //   after Prometheus restarts.
            // - using window or step as the duration. It results in too small rate() values for the first
            //   points of time series.
            //
            // So just return nan
            return nan;
        }
        prevValue = values[0]
        prevTimestamp = timestamps[0]
    } else if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0;
    }
    let vEnd = values[values.len() - 1];
    let tEnd = timestamps[timestamps.len() - 1];
    dv = vEnd - prevValue;
    dt = f64(tEnd - prevTimestamp) / 1e3;
    return dv / dt;
}

fn rollupIderiv(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    let timestamps = rfa.timestamps;
    if values.len() < 2 {
        if values.len() == 0 {
            return nan;
        }
        if rfa.prev_value.is_nan() {
            // It is impossible to determine the duration during which the value changed
            // from 0 to the current value.
            // The following attempts didn't work well:
            // - using scrape interval as the duration. It fails on Prometheus restarts when it
            //   skips scraping for the counter. This results in too high rate() value for the first point
            //   after Prometheus restarts.
            // - using window or step as the duration. It results in too small rate() values for the first
            //   points of time series.
            //
            // So just return nan
            return nan;
        }
        return (values[0] - rfa.prev_value) / (f64(timestamps[0] - rfa.prev_timestamp) / 1e3);
    }
    let vEnd = values[values.len() - 1];
    let tEnd = timestamps[timestamps.len() - 1];
    let values = values[0..values.len() - 1];
    let timestamps = timestamps[0..timestamps.len() - 1];
    // Skip data points with duplicate timestamps.
    while timestamps.len() > 0 && timestamps[timestamps.len() - 1] >= tEnd {
        timestamps = timestamps[0..timestamps.len() - 1];
    }
    let mut tStart: i64;
    let mut vStart: f64;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return 0;
        }
        tStart = rfa.prev_timestamp
        vStart = rfa.prev_value
    } else {
        tStart = timestamps[timestamps.len() - 1];
        vStart = values[timestamps.len() - 1];
    }
    dv = vEnd - vStart;
    dt = tEnd - tStart;
    return dv / (f64(dt) / 1e3);
}

fn rollupLifetime(rfa: RollupFuncArg) -> f64 {
    // Calculate the duration between the first and the last data points.
    timestamps = rfa.timestamps;
    if rfa.prev_value.is_nan() {
        if timestamps.len() < 2 {
            return nan;
        }
        return f64(timestamps[timestamps.len() - 1] - timestamps[0]) / 1e3;
    }
    if timestamps.len() == 0 {
        return nan;
    }
    return f64(timestamps[timestamps.len() - 1] - rfa.prev_timestamp) / 1e3;
}

fn rollupLag(rfa: RollupFuncArg) -> f64 {
// Calculate the duration between the current timestamp and the last data point.
    let timestamps = rfa.timestamps
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return (rfa.curr_timestamp - rfa.prev_timestamp) / 1e3;
    }
    return (rfa.curr_timestamp - timestamps[timestamps.len() - 1]) / 1e3;
}

fn rollupScrapeInterval(rfa: RollupFuncArg) -> f64 {
    // Calculate the average interval between data points.
    let timestamps = rfa.timestamps;
    if rfa.prev_value.is_nan() {
        if timestamps.len() < 2 {
            return nan;
        }
        return (f64(timestamps[timestamps.len() - 1] - timestamps[0]) / 1e3) / f64(timestamps.len() - 1);
    }
    if timestamps.len() == 0 {
        return nan;
    }
    return (f64(timestamps[timestamps.len() - 1] - rfa.prev_timestamp) / 1e3) / f64(timestamps.len());
}

fn rollupChangesPrometheus(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    // Do not take into account rfa.prev_value like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if values.len() < 1 {
        return nan;
    }
    prevValue = values[0];
    let mut n = 0;
    for v in values[1..] {
        if v != prevValue {
            n = n + 1;
            prevValue = v
        }
    }
    return n as f64;
}

fn rollupChanges(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    let mut prevValue = rfa.prev_value;
    let mut n = 0;
    if prevValue.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prevValue = values[0];
        values = values[1..];
        n = n + 1;
    }
    for v in values {
        if v != prevValue {
            n = n + 1;
            prevValue = v
        }
    }
    return f64(n);
}

fn rollupIncreases(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0;
    }
    let mut prevValue = rfa.prev_value;
    if prevValue.is_nan() {
        prevValue = values[0];
        values = values[1..];
    }
    if values.len() == 0 {
        return 0;
    }
    n = 0;
    for v in values {
        if v > prevValue {
            n = n + 1;
        }
        prevValue = v;
    }
    return f64(n);
}

// `decreases_over_time` logic is the same as `resets` logic.
const rollupDecreases: Rollupfunc = rollupResets;

fn rollupResets(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0;
    }
    let mut prev_value = rfa.prev_value;
    if math.IsNaN(prev_value) {
        prev_value = values[0];
        values = values[1..];
    }
    if values.len() == 0 {
        return 0;
    }
    let mut n = 0;
    for v in values {
        if v < prev_value {
            n = n + 1;
        }
        prev_value = v;
    }
    return n;
}


// getCandlestickValues returns a subset of rfa.values suitable for rollup_candlestick
//
// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309 for details.
fn getCandlestickValues(rfa: RollupFuncArg) -> Option<Vec<f64>> {
    let curr_timestamp = rfa.curr_timestamp;
    let mut timestamps = rfa.timestamps;
    while timestamps.len() > 0 && timestamps[timestamps.len() - 1] >= curr_timestamp {
        timestamps = &timestamps[0..timestamps.len() - 1]
    }
    if timestamps.len() == 0 {
        return None;
    }
    return rfa.values[0..timestamps.len()];
}

fn getFirstValueForCandlestick(rfa: RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    return nan;
}

fn rollupOpen(rfa: RollupFuncArg) -> f64 {
    let v = getFirstValueForCandlestick(rfa);
    if !v.is_nan() {
        return v;
    }
    let values = getCandlestickValues(rfa);
    if values.len() == 0 {
        return nan;
    }
    return values[0];
}

fn rollupClose(rfa: RollupFuncArg) -> f64 {
    let values = getCandlestickValues(rfa);
    if values.len() == 0 {
        return getFirstValueForCandlestick(rfa);
    }
    return values.last().unwrap();
}

fn rollupHigh(rfa: RollupFuncArg) -> f64 {
    let mut values = getCandlestickValues(rfa)
    let max = getFirstValueForCandlestick(rfa)
    if math.IsNaN(max) {
        if values.len() == 0 {
            return nan;
        }
        max = values[0];
        values = values[1..];
    }
    for v in values {
        if v > max {
            max = v
        }
    }
    return max;
}

fn rollupLow(rfa: RollupFuncArg) -> f64 {
    let mut values = getCandlestickValues(rfa);
    let min = getFirstValueForCandlestick(rfa);
    if min.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        min = values[0]
        values = values[1..];
    }
    for v in values {
        if v < min {
            min = v
        }
    }
    return min;
}


fn rollupModeOverTime(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // Copy rfa.values to a.A, since modeNoNaNs modifies a.A contents.
    let a = getf64s()
    a.A = append(a.A[0..0], rfa.values...)
    let result = modeNoNaNs(rfa.prev_value, a.A)
    putf64s(a);
    return result;
}

fn rollupAscentOverTime(rfs: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let mut prev_value = rfa.prevValue;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        values = values[1..];
    }
    let s: f64 = 0;
    for v in values {
        let d = v - prev_value;
        if d > 0 {
            s = s + d
        }
        prev_value = v
    }
    return s;
}

fn rollupDescentOverTime(rfs: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let prev_value = rfa.prevValue;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        values = values[1..]
    }
    let mut s: f64 = 0;
    for v in values {
        let d = prev_value - v;
        if d > 0 {
            s += d
        }
        prev_value = v
    }
    return s;
}

fn rollupZScoreOverTime(rfs: RollupFuncArg) -> f64 {
// See https://about.gitlab.com/blog/2019/07/23/anomaly-detection-using-prometheus/#using-z-score-for-anomaly-detection
    let scrapeInterval = rollupScrapeInterval(rfa)
    let lag = rollupLag(rfa)
    if scrapeInterval.is_nan() || lag.is_nan() || lag > scrapeInterval {
        return nan;
    }
    let d = rollupLast(rfa) - rollupAvg(rfa);
    if d == 0 {
        return 0;
    }
    return d / rollupStddev(rfa);
}

fn rollupFirst(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return values[0];
}

fn rollupDefault(rfa: RollupFuncArg) -> f64 {
    let values = rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    // Intentionally do not skip the possible last Prometheus staleness mark.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1526 .
    return values.last().unwrap();
}

fn rollupLast(rfa: RollupFuncArg) -> f64 {
    let values = rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return values.last().unwrap();
}

fn rollupDistinct(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0;
    }
    // todo: use hashbrown for perf throughout
    let m = HashSet::from(values);
    return m.len() as f64;
}


fn rollupIntegrate(rfa: RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns. 
    let mut values = rfa.values;
    let mut timestamps = rfa.timestamps;
    let mut prev_value = rfa.prev_value;
    let prev_timestamp = rfa.curr_timestamp - rfa.window;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        prev_timestamp = timestamps[0];
        values = values[1..];
        timestamps = timestamps[1..];
    }
    let mut sum: f64 = 0.0;
    for (i, v) in values {
        let timestamp = timestamps[i];
        let dt = (timestamp - prev_timestamp) / 1e3;
        sum = sum + prev_value * dt;
        prev_timestamp = timestamp;
        prev_value = v;
    }
    dt = rfa.curr_timestamp - prev_timestamp) / 1e3
    sum = prev_value * dt;
    return sum;
}

fn rollupFake(rfa: RollupFuncArg) -> f64 {
    panic!("BUG: rollupFake shouldn't be called");
    return 0;
}

pub(super) fn get_scalar(arg: RollupArgValue, argNum: usize) -> Result<Vec<i64>, Error> {
    match arg {
        RollupArgValue::Timeseries(ts) => {
            if ts.len() != 1 {
                return nil;, fmt.Errorf(`arg # %d must contain a single timeseries;
                got % d
                timeseries`, argNum + 1, len(ts))
            }
            return ts[0].values;
        }
        _ => {
            return nil;, fmt.Errorf(`unexpected type for arg # %d;
            got % T;
            want % T`, argNum + 1, arg, ts)
        }
    }
}

fn getIntNumber(arg: RollupArgValue, argNum: usize) -> Result<i64, Error> {
    let v = get_scalar(arg, argNum)?;
    let mut n = 0;
    if v.len() > 0 {
        n = int(v[0])
    }
    return Ok(n);
}

fn getString(tss: Vec<Timeseries>, arg_num: usize) -> Result<String, Error> {
    if tss.len() != 1 {
        return "";, fmt.Errorf(`arg # %d must contain a single timeseries;
        got % d
        timeseries`, arg_num + 1, len(tss))
    }
    let ts = tss[0];
    for v in ts.values {
        if !v.is_nan() {
            return fmt.Errorf(`arg # %d contains non - string timeseries`, arg_num + 1);
        }
    }
    return Ok(ts.metric_name.metric_group());
}

#[inline]
fn expectRollupArgsNum(args: Vec<RollupArgValue>, expected_num: usize) {
    if args.len() == expected_num {
        return;
    }
    return fmt.Errorf(`unexpected number of args;
    got % d;
    want % d`, args.len(), expected_num)
}