use std::collections::HashSet;
use lib::error::*;
use phf::phf_map;
use phf::phf_set;
use metricsql::types::Expression;
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

pub type NewRollupFunc = fn(args: &Vec<RollupArgValue>) -> Result<RollupFunc>;

static ROLLUP_FUNCTIONS: phf::Map<&'static str, NewRollupFunc> = phf_map! {
	"absent_over_time":        newRollupFuncOneArg(rollupAbsent),
	"aggr_over_time":          newRollupFuncTwoArgs(rollupFake),
	"ascent_over_time":        newRollupFuncOneArg(rollupAscentOverTime),
	"avg_over_time":           newRollupFuncOneArg(rollupAvg),
	"changes":                 newRollupFuncOneArg(rollupChanges),
	"changes_prometheus":      newRollupFuncOneArg(rollupChangesPrometheus),
	"count_eq_over_time":      newRollupCountEQ,
	"count_gt_over_time":      newRollupCountGT,
	"count_le_over_time":      newRollupCountLE,
	"count_ne_over_time":      newRollupCountNE,
	"count_over_time":         newRollupFuncOneArg(rollupCount),
	"decreases_over_time":     newRollupFuncOneArg(rollupDecreases),
	"default_rollup":          newRollupFuncOneArg(rollupDefault), // default rollup func
	"delta":                   newRollupFuncOneArg(rollupDelta),
	"delta_prometheus":        newRollupFuncOneArg(rollupDeltaPrometheus),
	"deriv":                   newRollupFuncOneArg(rollupDerivSlow),
	"deriv_fast":              newRollupFuncOneArg(rollupDerivFast),
	"descent_over_time":       newRollupFuncOneArg(rollupDescentOverTime),
	"distinct_over_time":      newRollupFuncOneArg(rollupDistinct),
	"duration_over_time":      newRollupDurationOverTime,
	"first_over_time":          newRollupFuncOneArg(rollupFirst),
	"geomean_over_time":       newRollupFuncOneArg(rollupGeomean),
	"histogram_over_time":     newRollupFuncOneArg(rollupHistogram),
	"hoeffding_bound_lower":   newRollupHoeffdingBoundLower,
	"hoeffding_bound_upper":   newRollupHoeffdingBoundUpper,
	"holt_winters":            newRollupHoltWinters,
	"idelta":                  newRollupFuncOneArg(rollupIdelta),
	"ideriv":                  newRollupFuncOneArg(rollupIderiv),
	"increase":                newRollupFuncOneArg(rollupDelta),           // + rollupFuncsRemoveCounterResets
	"increase_prometheus":     newRollupFuncOneArg(rollupDeltaPrometheus), // + rollupFuncsRemoveCounterResets
	"increase_pure":           newRollupFuncOneArg(rollupIncreasePure),    // + rollupFuncsRemoveCounterResets
	"increases_over_time":     newRollupFuncOneArg(rollupIncreases),
	"integrate":               newRollupFuncOneArg(rollupIntegrate),
	"irate":                   newRollupFuncOneArg(rollupIderiv), // + rollupFuncsRemoveCounterResets
	"lag":                     newRollupFuncOneArg(rollupLag),
	"last_over_time":          newRollupFuncOneArg(rollupLast),
	"lifetime":                newRollupFuncOneArg(rollupLifetime),
	"max_over_time":           newRollupFuncOneArg(rollupMax),
	"min_over_time":           newRollupFuncOneArg(rollupMin),
	"mode_over_time":          newRollupFuncOneArg(rollupModeOverTime),
	"predict_linear":          newRollupPredictLinear,
	"present_over_time":       newRollupFuncOneArg(rollupPresent),
	"quantile_over_time":      newRollupQuantile,
	"quantiles_over_time":     newRollupQuantiles,
	"range_over_time":         newRollupFuncOneArg(rollupRange),
	"rate":                    newRollupFuncOneArg(rollupDerivFast), // + rollupFuncsRemoveCounterResets
	"rate_over_sum":           newRollupFuncOneArg(rollupRateOverSum),
	"resets":                  newRollupFuncOneArg(rollupResets),
	"rollup":                  newRollupFuncOneArg(rollupFake),
	"rollup_candlestick":      newRollupFuncOneArg(rollupFake),
	"rollup_delta":            newRollupFuncOneArg(rollupFake),
	"rollup_deriv":            newRollupFuncOneArg(rollupFake),
	"rollup_increase":         newRollupFuncOneArg(rollupFake), // + rollupFuncsRemoveCounterResets
	"rollup_rate":             newRollupFuncOneArg(rollupFake), // + rollupFuncsRemoveCounterResets
	"rollup_scrape_interval":  newRollupFuncOneArg(rollupFake),
	"scrape_interval":         newRollupFuncOneArg(rollupScrapeInterval),
	"share_gt_over_time":      newRollupShareGT,
	"share_le_over_time":      newRollupShareLE,
	"stale_samples_over_time": newRollupFuncOneArg(rollupStaleSamples),
	"stddev_over_time":        newRollupFuncOneArg(rollupStddev),
	"stdvar_over_time":        newRollupFuncOneArg(rollupStdvar),
	"sum_over_time":           newRollupFuncOneArg(rollupSum),
	"sum2_over_time":          newRollupFuncOneArg(rollupSum2),
	"tfirst_over_time":        newRollupFuncOneArg(rollupTfirst),
	// `timestamp` function must return timestamp for the last datapoint on the current window
	// in order to properly handle offset and timestamps unaligned to the current step.
	// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415 for details.
	"timestamp":              newRollupFuncOneArg(rollupTlast),
	"timestamp_with_name":    newRollupFuncOneArg(rollupTlast), // + rollupFuncsKeepMetricName
	"tlast_change_over_time": newRollupFuncOneArg(rollupTlastChange),
	"tlast_over_time":        newRollupFuncOneArg(rollupTlast),
	"tmax_over_time":         newRollupFuncOneArg(rollupTmax),
	"tmin_over_time":         newRollupFuncOneArg(rollupTmin),
	"zscore_over_time":       newRollupFuncOneArg(rollupZScoreOverTime),
};


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
        self.prev_value = 0.0;
        self.prev_timestamp = 0;
        self.values = vec![];
        self.timestamps = vec![];
        self.curr_timestamp = 0;
        self.idx = 0;
        self.window = 0;
        self.tsm = nil
    }
}

// RollupFunc must return rollup value for the given rfa.
//
// prev_value may be nan, values and timestamps may be empty.
pub(crate) type RollupFunc = fn(rfa: &RollupFuncArg) -> f64;

static ROLLUP_AGGR_FUNCTIONS: phf::Map<&'static str, NewRollupFunc> = phf_map! {
    "absent_over_time":        rollupAbsent,
	"ascent_over_time":        rollupAscentOverTime,
	"avg_over_time":           rollupAvg,
	"changes":                 rollupChanges,
	"count_over_time":         rollupCount,
	"decreases_over_time":     rollupDecreases,
	"default_rollup":          rollupDefault,
	"delta":                   rollupDelta,
	"deriv":                   rollupDerivSlow,
	"deriv_fast":              rollupDerivFast,
	"descent_over_time":       rollupDescentOverTime,
	"distinct_over_time":      rollupDistinct,
	"first_over_time":          rollupFirst,
	"geomean_over_time":       rollupGeomean,
	"idelta":                  rollupIdelta,
	"ideriv":                  rollupIderiv,
	"increase":                rollupDelta,
	"increase_pure":           rollupIncreasePure,
	"increases_over_time":     rollupIncreases,
	"integrate":               rollupIntegrate,
	"irate":                   rollupIderiv,
	"lag":                     rollupLag,
	"last_over_time":          rollupLast,
	"lifetime":                rollupLifetime,
	"max_over_time":           rollupMax,
	"min_over_time":           rollupMin,
	"mode_over_time":          rollupModeOverTime,
	"present_over_time":       rollupPresent,
	"range_over_time":         rollupRange,
	"rate":                    rollupDerivFast,
	"rate_over_sum":           rollupRateOverSum,
	"resets":                  rollupResets,
	"scrape_interval":         rollupScrapeInterval,
	"stale_samples_over_time": rollupStaleSamples,
	"stddev_over_time":        rollupStddev,
	"stdvar_over_time":        rollupStdvar,
	"sum_over_time":           rollupSum,
	"sum2_over_time":          rollupSum2,
	"tfirst_over_time":         rollupTfirst,
	"timestamp":               rollupTlast,
	"timestamp_with_name":     rollupTlast,
	"tlast_change_over_time":  rollupTlastChange,
	"tlast_over_time":         rollupTlast,
	"tmax_over_time":          rollupTmax,
	"tmin_over_time":          rollupTmin,
	"zscore_over_time":        rollupZScoreOverTime,
};

// VictoriaMetrics can increase lookbehind window in square brackets for these functions
// if the given window doesn't contain enough samples for calculations.
//
// This is needed in order to return the expected non-empty graphs when zooming in the graph in Grafana,
// which is built with `func_name(metric[$__interval])` query.
static ROLLUP_FUNCTIONS_CAN_ADJUST_WINDOW: phf::Set<&'static str> = phf_set! {
    "default_rollup",
	"deriv",
	"deriv_fast",
	"ideriv",
	"irate",
	"rate",
	"rate_over_sum",
	"rollup",
	"rollup_candlestick",
	"rollup_deriv",
	"rollup_rate",
	"rollup_scrape_interval",
	"scrape_interval",
	"timestamp",
};

static ROLLUP_FUNCTIONS_REMOVE_COUNTER_RESETS: phf::Set<&'static str> = phf_set! {
	"increase",
	"increase_prometheus",
	"increase_pure",
	"irate",
	"rate",
	"rollup_increase",
	"rollup_rate"
};

// These functions don't change physical meaning of input time series,
// so they don't drop metric name
static ROLLUP_FUNCTIONS_KEEP_METRIC_NAME: phf::Set<&'static str> = phf_set! {
    "avg_over_time",
	"default_rollup",
	"first_over_time",
	"geomean_over_time",
	"hoeffding_bound_lower",
	"hoeffding_bound_upper",
	"holt_winters",
	"last_over_time",
	"max_over_time",
	"min_over_time",
	"mode_over_time",
	"predict_linear",
	"quantile_over_time",
	"quantiles_over_time",
	"rollup",
	"rollup_candlestick",
	"timestamp_with_name",
};

pub(crate) fn get_rollup_func(func_name: &str) -> Option<&NewRollupFunc> {
    let lower = func_name.to_lowercase().as_str();
    return ROLLUP_FUNCTIONS.get(lower);
}

// todo: use in HIR
pub(crate) fn get_rollup_aggr_func_names(expr: &Expression) -> Result<Vec<String>> {
    let fe = match expr {
        Expression::Aggregation(afe) => {
            // This is for incremental aggregate function case:
            //
            //     sum(aggr_over_time(...))
            //
            // See aggr_incremental.go for details.
            let _expr = &afe.args[0];
            match _expr {
                Expression::Function(f) => f,
                _ => {
                    None
                }
            }
        },
        Expression::Function(fe) => {
            fe
        },
        _ => {
            None
        }
    };

    if fe.is_none() {
        let msg = format!("BUG: unexpected expression; want metricsql.FuncExpr; got {}; value: {}", expr.kind(), expr);
        return Err(Error::new(msg));
    }

    if fe.name != "aggr_over_time" {
        let msg = format!("BUG: unexpected function name: {}; want `aggr_over_time`", fe.name);
        return Err(Error::new(msg));
    }

    if fe.args.len() != 2 {
        let msg = format!("unexpected number of args to aggr_over_time(); got {}; want {}", fe.args.len(), 2);
        return Err(Error::new(msg));
    }

    let arg = &fe.args[0];
    let mut aggr_func_names: Vec<String> = Vec::with_capacity(1);
    match arg {
        Expression::String(se) => {
            aggr_func_names.push(se.s.as_string());
        },
        Expression::Function(fe) => {
            if fe.name.len() > 0 {
                let msg = format!("{} cannot be passed to aggr_over_time(); expecting quoted aggregate function name",
                                       arg);
                return Err(Error::new(msg));
            }
            for exp in fe.args {
                match exp {
                    Expression::String(se) => {
                        let name = &se.s;
                        if ROLLUP_AGGR_FUNCTIONS.get(&name).is_some() {
                            aggr_func_names.push(se.s);
                        } else {
                            let msg = format!("{} cannot be used in `aggr_over_time` function; expecting quoted aggregate function name", name);
                            return Err(Error::new(msg));
                        }
                    },
                    _ => {
                        let msg = format!("{} cannot be passed here; expecting quoted aggregate function name", e);
                        return Err(Error::new(msg));
                    }
                }
            }
        },
        _ => {
            let msg = format!("{} cannot be passed to aggr_over_time(); expecting a list of quoted aggregate function names",
                                                         arg);
            return Err(Error::new(msg));
        }
    }

    return Ok(aggr_func_names)
}

type PreFunc = fn(values: &[f64], timestamps: &[i64]) -> ();

fn get_rollup_configs(
    name: &str,
    rf: RollupFunc,
    expr: &Expression,
    start: i64, end: i64, step: i64, window: i64,
    lookback_delta: i64,
    shared_timestamps: &[i64]) -> Result<(PreFunc, Vec<RollupConfig>)> {

    let mut pre_func: PreFunc = |values, timestamps| {};

    if ROLLUP_FUNCTIONS_REMOVE_COUNTER_RESETS.contains(name) {
        pre_func = |values, timestamps| {
            removeCounterResets(values)
        }
    }
    let new_rollup_config = |rf: RollupFunc, tag_value: &str| -> RollupConfig {
        return RollupConfig{
            tag_value,
            func:            rf,
            start,
            end,
            step,
            window,
            may_adjust_window:  ROLLUP_FUNCTIONS_CAN_ADJUST_WINDOW.contains(name),
            lookback_delta,
            timestamps: shared_timestamps,
            is_default_rollup: name == "default_rollup",
        }
    };
    
    let append_rollup_configs = |mut dst: &Vec<RollupConfig>| {
        dst.push(new_rollup_config(rollupMin, "min"));
        dst.push(new_rollup_config(rollupMax, "max"));
        dst.push(new_rollup_config(rollupAvg, "avg"));
    };

    let mut rcs: Vec<RollupConfig> = Vec::new();
    match name {
        "rollup" => {
            append_rollup_configs(&rcs);
        },
        "rollup_rate" | "rollup_deriv" => {
            let mut pre_func_prev = pre_func;
            pre_func = |values, timestamps| {
                pre_func_prev(values, timestamps);
                derivValues(values, timestamps)
            };
            append_rollup_configs(&mut rcs);
        }
        "rollup_increase" | "rollup_delta" => {
            let mut pre_func_prev = pre_func;
            pre_func = |values, timestamps| {
                pre_func_prev(values, timestamps);
                deltaValues(values)
            };
            append_rollup_configs(&rcs);
        }
        "rollup_candlestick" => {
            rcs.push(new_rollup_config(rollupOpen, "open"));
            rcs.push(new_rollup_config(rollupClose, "close"));
            rcs.push(new_rollup_config(rollupLow, "low"));
            rcs.push(new_rollup_config(rollupHigh, "high"));
        },
        "rollup_scrape_interval" => {
            let pre_func_prev = pre_func;
            pre_func = |mut values, timestamps| {
                pre_func_prev(values, timestamps);
                // Calculate intervals in seconds between samples.
                let mut ts_secs_prev = nan;
                for (i, ts) in timestamps.iter().enumerate() {
                    let ts_secs = ts / 1000;
                    values[i] = ts_secs - ts_secs_prev;
                    tsSecsPrev: ts_secs_prev = ts_secs;
                };
                if values.len() > 1 {
                    // Overwrite the first NaN interval with the second interval,
                    // So min, max and avg rollups could be calculated properly, since they don't expect to receive NaNs.
                    values[0] = values[1]
                }
            };
            appendRollupConfigs(rcs);
        }
        "aggr_over_time" => {
            let aggr_func_names = get_rollup_aggr_func_names(expr);
            if aggr_func_names.is_error() {
                return nil, nil, fmt.Errorf("invalid args to %s: %w", expr, err)
            }
            for aggrFuncName in aggr_func_names {
                if ROLLUP_FUNCTIONS_REMOVE_COUNTER_RESETS.contains(aggrFuncName) {
                    // There is no need to save the previous pre_func, since it is either empty or the same.
                    pre_func = |values, timestamps| {
                        removeCounterResets(values)
                    }
                }
                let rf = rollupAggrFuncs[aggrFuncName]
                rcs.push(new_rollup_config(rf, aggrFuncName));
            }
        },
        _ => {
            rcs.push(new_rollup_config(rf, ""));
        }
    }
    Ok((preFunc, rcs))
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub(crate) struct RollupConfig {
    // This tag value must be added to "rollup" tag if non-empty.
    tag_value: String,
    func: RollupFunc,
    start: i64,
    end: i64,
    step: i64,
    window: i64,

    // Whether window may be adjusted to 2 x interval between data points.
    // This is needed for functions which have dt in the denominator
    // such as rate, deriv, etc.
    // Without the adjustment their value would jump in unexpected directions
    // when using window smaller than 2 x scrape_interval.
    may_adjust_window: bool,

    timestamps: Vec<i64>,

    // LoookbackDelta is the analog to `-query.lookback-delta` from Prometheus world.
    lookback_delta: i64,

    // Whether default_rollup is used.
    is_default_rollup: bool,
}

impl RollupConfig {
    // Do calculates rollups for the given timestamps and values, appends
    // them to dst_values and returns results.
    //
    // rc.Timestamps are used as timestamps for dst_values.
    //
    // timestamps must cover time range [rc.start - rc.Window - maxSilenceInterval ... rc.end].
    //
    // Do cannot be called from concurrent goroutines.
    pub(crate) fn Do(&self, mut dst_values: &Vec<i64>, values: &[i64], timestamps: &[i64]) -> Vec<f64> {
        return rc.doInternal(dst_values, None, values, timestamps)
    }

    fn do_internal(&self, mut dst_values: &[i64], values: &[i64], timestamps: &[i64]) -> Vec<f64> {
        // Sanity checks.
        if self.step <= 0 {
            let msg = format!("BUG: Step must be bigger than 0; got {}", rc.step);
            return Err(Error::new(msg));
        }
        if self.start > rc.end {
            let msg = format!("BUG: start cannot exceed End; got {} vs {}", rc.start, rc.end);
            return Err(Error::new(msg));
        }
        if self.window < 0 {
            let msg = format!("BUG: Window must be non-negative; got {}", rc.Window);
            return Err(Error::new(msg));
        }
        match validateMaxPointsPerTimeseries(rc.start, rc.end, rc.step) {
            Err(err) => {
                let msg = format!("BUG: %s; this must be validated before the call to rollupConfig.Do", err);
                return Err(Error::new(msg))
            },
            _ => ();
        }

        // Extend dst_values in order to remove mallocs below.
        let dstValues = decimal.ExtendFloat64sCapacity(dst_values, rc.timestamps.len());

        let scrape_interval = get_scrape_interval(&timestamps);
        let mut max_prev_interval = get_max_prev_interval(scrape_interval);
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
        if self.may_adjust_window && window < max_prev_interval {
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
            let mut t_start = tEnd - window;
            let ni = seek_first_timestamp_idx_after(&timestamps[i..], t_start, ni);
            i = i + ni;
            if j < i {
                j = i
            }
            let nj = seek_first_timestamp_idx_after(&timestamps[j..], tEnd, nj);
            j += nj;

            rfa.prevValue = nan;
            rfa.prevTimestamp = t_start - max_prev_interval;
            if i < timestamps.len() && i > 0 && timestamps[i-1] > rfa.prevTimestamp {
                rfa.prevValue = values[i-1];
                rfa.prevTimestamp = timestamps[i-1].clone();
            }
            rfa.values = values[i..j];
            rfa.timestamps = &timestamps[i..j];
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


fn new_rollup_func_one_arg(rf: &RollupFunc) -> NewRollupFunc {
    |args: &[RollupArgValue]| -> Result<RollupFunc> {
        expect_rollup_args_num(args, 1)?;
        Ok(rf)
    }
}

fn new_rollup_func_two_args(rf: RollupFunc) -> NewRollupFunc {
    |args: &[RollupArgValue]| -> Result<RollupFunc> {
        expect_rollup_args_num(&args, 2)?;
        Ok(rf)
    }
}

fn seek_first_timestamp_idx_after(timestamps: &[i64], seek_timestamp: i64, n_hint: usize) -> usize {
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
    let i = binary_search_int64(ts, seek_timestamp + 1);
    return start_idx + i;
}

fn binary_search_int64(a: &[i64], v: i64) -> usize {
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


fn get_scrape_interval(timestamps: &[i64]) -> i64 {
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
    let scrape_interval = quantile(0.6, intervals) as i64;
    a.A = intervals;
    putFloat64s(a);
    if scrape_interval <= 0 {
        return int64(maxSilenceInterval);
    }
    return scrape_interval;
}

fn get_max_prev_interval(scrape_interval: &i64) -> i64 {
    // Increase scrape_interval more for smaller scrape intervals in order to hide possible gaps
    // when high jitter is present.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/139 .
    if scrape_interval <= 2 * 1000 {
        return scrape_interval + 4 * scrape_interval;
    }
    if scrape_interval <= 4 * 1000 {
        return scrape_interval + 2 * scrape_interval;
    }
    if scrape_interval <= (8 * 1000) as i64 {
        return scrape_interval + scrape_interval;
    }
    if scrape_interval <= 16 * 1000 {
        return scrape_interval + scrape_interval / 2;
    }
    if scrape_interval <= 32 * 1000 {
        return scrape_interval + scrape_interval / 4;
    }
    return scrape_interval + scrape_interval / 8;
}

fn remove_counter_resets(values: &mut Vec<f64>) {
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
                *v = prev_value;
            } else {
                correction += prev_value;
            }
        }
        prev_value = *v;
        values[i] = v + correction;
        i = i + 1;
    }
}

fn delta_values(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from vmstorage.
    if values.len() == 0 {
        return;
    }
    let mut prev_delta: f64 = 0.0;
    let mut prev_value = values[0];
    for (i, v) in values[1..].iter().enumerate() {
        prev_delta = v - prev_value;
        values[i] = prev_delta;
        prev_value = *v;
    }
    values[values.len() - 1] = prev_delta
}

fn deriv_values(values: &mut [f64], timestamps: &Vec<i64>) {
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

fn new_rollup_holt_winters(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    expect_rollup_args_num(args, 3)?;
    let sfs = get_scalar(&args[1], 1)?;
    let tfs = get_scalar(&args[2], 2)?;

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
            values = &values[1..];
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

fn new_rollup_predict_linear(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    expect_rollup_args_num(args, 2)?;
    let secs = get_scalar(&args[1], 1)?;

    let f = |rfa: RollupFuncArg| -> f64 {
        let (v, k) = linear_regression(rfa);
        if v.is_nan() {
            return nan;
        }
        sec = secs[rfa.idx];
        return v + k * sec;
    };
    
    Ok(f)
}

fn linear_regression(rfa: &RollupFuncArg) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
    let n = values.len();
    if n == 0 {
        return (nan, nan);
    }
    if are_const_values(values) {
        return (values[0], 0.0);
    }

    // See https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    let intercept_time = &rfa.curr_timestamp;
    let mut v_sum: f64 = 0.0;
    let mut t_sum: f64 = 0.0;
    let mut tv_sum: f64 = 0.0;
    let mut tt_sum: f64 = 0.0;
    for (i, v) in values.iter().enumerate() {
        let dt = (timestamps[i] - intercept_time) / 1e3;
        v_sum += v;
        t_sum += dt;
        tv_sum += dt * v;
        tt_sum += dt * dt
    }
    let mut k: f64 = 0.0;
    let t_diff = tt_sum - t_sum * t_sum / n;
    if t_diff.abs() >= 1e-6 {
        // Prevent from incorrect division for too small t_diff values.
        k = (tv_sum - t_sum * v_sum / n) / t_diff;
    }
    let v = v_sum / n - k * t_sum / n;
    return (v, k);
}

fn are_const_values(values: &Vec<f64>) -> bool {
    if values.len() <= 1 {
        return true;
    }
    let mut v_prev = values[0];
    for v in values[1..] {
        if v != v_prev {
            return false;
        }
        v_prev = v
    }
    return true;
}

fn new_rollup_duration_over_time(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let d_maxs = get_scalar(&args[1], 1)?;

   let f = move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut timestamps = rfa.timestamps;
        if timestamps.len() == 0 {
            return nan;
        }
        let mut t_prev = timestamps[0];
        let mut d_sum: i64 = 0;
        let d_max = (d_maxs[rfa.idx] * 1000) as i64;
        for t in timestamps.iter() {
            let d = &t - t_prev;
            if d <= d_max {
                d_sum += d;
            }
            t_prev = t
        }
        return (d_sum / 1000) as f64
    };
    
    Ok(f)
}

fn new_rollup_share_le(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return new_rollup_share_filter(args, count_filter_le);
}

fn count_filter_le(values: &Vec<f64>, le: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if *v <= le {
            n = n + 1;
        }
    }
    return n;
}

fn new_rollup_share_gt(args: Vec<RollupArgValue>) -> Result<RollupFunc, Error> {
    return new_rollup_share_filter(args, count_filter_gt);
}

#[inline]
fn count_filter_gt(values: &[f64], gt: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if *v > gt {
            n = n + 1;
        }
    }
    return n;
}

#[inline]
fn count_filter_eq(values: &[f64], eq: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if v == eq {
            n = n + 1;
        }
    }
    return n;
}

#[inline]
fn count_filter_ne(values: &[f64], ne: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if v != ne {
            n = n + 1;
        }
    }
    return n;
}

fn new_rollup_share_filter(args: Vec<RollupArgValue>, count_filter: fn(values: &[f64], limit: f64) -> i32) -> Result<RollupFunc> {
    let rf = new_rollup_count_filter(args, count_filter)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let n = rf(&rfa);
        return n / rfa.values.len();
    };
    Ok(f)
}

fn new_rollup_count_le(args: Vec<RollupArgValue>) -> Result<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_le);
}

fn new_rollup_count_gt(args: Vec<RollupArgValue>) -> Result<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_gt);
}

fn new_rollup_count_eq(args: Vec<RollupArgValue>) -> Result<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_eq);
}

fn new_rollup_count_ne(args: Vec<RollupArgValue>) -> Result<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_ne);
}

fn new_rollup_count_filter(args: Vec<RollupArgValue>, count_filter: fn(values: &[f64], limit: f64) -> i32) -> Result<RollupFunc> {
    expect_rollup_args_num(&args, 2)?;
    let limits = get_scalar(&args[1], 1)?;
    move |rfa: RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        if values.len() == 0 {
            return nan;
        }
        let limit = limits[rfa.idx];
        return count_filter(&values, limit) as f64;
    }
}

fn new_rollup_hoeffding_bound_lower(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(&args[0], 0)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(rfa, phis);
        return avg - bound;
    };

    Ok(f)
}

fn new_rollup_hoeffding_bound_upper(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(&args[0], 0)?;
    let f = move |rfa: RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(&rfa, &phis);
        return avg + bound;
    };

    Ok(f)
}

fn rollup_hoeffding_bound_internal(rfa: &RollupFuncArg, phis: &[f64]) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        return (nan, nan);
    }
    if values.len() == 1 {
        return (0.0, values[0]);
    }
    let v_max = rollup_max(rfa);
    let v_min = rollup_min(rfa);
    let v_avg = rollup_avg(rfa);
    let v_range = v_max - v_min;
    if v_range <= 0.0 {
        return (0.0, v_avg);
    }
    let phi = phis[rfa.idx];
    if phi >= 1.0 {
        return (inf, v_avg);
    }
    if phi <= 0.0 {
        return (0.0, v_avg);
    }
// See https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
// and https://www.youtube.com/watch?v=6UwcqiNsZ8U&feature=youtu.be&t=1237
    let bound = v_range * math.Sqrt(math.Log(1 / (1 - phi)) / (2 * float64(values.len())))
    return (bound, v_avg)
}

fn new_rollup_quantiles(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    if args.len() < 3 {
        let msg = format!("unexpected number of args: {}; want at least 3 args", args.len());
        return Err(Error::new(msg));
    }
    tssPhi, ok = args[0].([] * timeseries)
    if !ok {
        return nil;, fmt.Errorf("unexpected type for phi arg: %T; want string", args[0])
    }
    let phi_label = get_string(tssPhi, 0)?;
    let phi_args = args[1 .. args.len() - 1];
    let phis = Vec::with_capacity(phi_args.len());
    // todo: smallvec ??
    let phi_strs: Vec<String> = Vec::with_capacity(phi_args.len());

    for (i, phiArg) in phi_args {
        let phi_values = get_scalar(phiArg, i + 1);
        if !Some(phi_values) {
            fmt.Errorf("cannot obtain phi from arg #{}: %w", i + 1, err)
        }
        phis[i] = phi_values[0];
        phi_strs[i] = format!("{}", phi_values[0]);
    }
    let f = move |rfa: &RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = &rfa.values;
        if values.len() == 0 {
            return rfa.prev_value;
        }
        if values.len() == 1 {
            // Fast path - only a single value.
            return values[0];
        }
        let qs = getFloat64s();
        qs.A = quantiles(qs.A[0..0], phis, values);
        let idx = rfa.idx;
        let tsm = rfa.tsm;
        for (i, phiStr) in phi_strs {
            let ts = tsm.GetOrCreateTimeseries(phi_label, phiStr);
            ts.values[idx] = qs.A[i]
        }
        putFloat64s(qs);
        return nan;
    }

    Ok(f)
}

fn new_rollup_quantile(args: &Vec<RollupArgValue>) -> Result<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(&args[0], 0)?;
    |rfa: RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = rfa.values;
        let phi = phis[rfa.idx];
        quantile(phi, values);
    };
    return rf;
}

fn rollup_histogram(rfa: &RollupFuncArg) -> f64 {
    let mut values = rfa.values;
    let tsm = rfa.tsm;
    tsm.h.Reset();
    for v in values {
        tsm.h.Update(v)
    }
    idx = rfa.idx;
    tsm.h.VisitNonZeroBuckets( fn (vmrange
    string, count
    uint64) {
        ts = tsm.GetOrCreateTimeseries("vmrange", vmrange)
        ts.values[idx] = f64(count)
    })
    return nan;
}

fn rollup_avg(rfa: &RollupFuncArg) -> f64 {
// Do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
// since it is slower and has no significant benefits in precision.

// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    let sum: f64 = values.iter().fold(0.0,|r, x| r + *x);
    return sum / values.len();
}

fn rollup_min(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    minValue = values[0];
    for v in values.iter() {
        if v < minValue {
            minValue = v
        }
    }
    return minValue;
}

fn rollup_max(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_tmin(rfa: &RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = rfa.values;
    let mut timestamps = rfa.timestamps;
    if values.len() == 0 {
        return nan;
    }
    minValue = values[0];
    minTimestamp = timestamps[0];
    for (i, v) in values.iter().enumerate() {
        // Get the last timestamp for the minimum value as most users expect.
        if v <= minValue {
            minValue = v;
            minTimestamp = timestamps[i];
        }
    }
    return (minTimestamp / 1e3) as f64;
}

fn rollup_tmax(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    let mut timestamps = rfa.timestamps;
    if values.len() == 0 {
        return nan;
    }
    let mut max_value = values[0];
    let mut max_timestamp = timestamps[0];
    for (i, v) in values.iter().enumerate() {
        // Get the last timestamp for the maximum value as most users expect.
        if *v >= max_value {
            max_value = *v;
            max_timestamp = timestamps[i];
        }
    }
    return (max_timestamp / 1e3) as f64;
}
///////////////////////

fn rollup_tfirst(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_tlast(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_tlast_change(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_sum(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    values.iter().fold(0.0,|r, x| r + *x)
}

fn rollup_rate_over_sum(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change since rfa.prev_value.
        return 0.0;
    }
    dt = rfa.window;
    if !rfa.prev_value.is_nan() {
        dt = timestamps[timestamps.len() - 1] - rfa.prev_timestamp
    }
    let sum = rfa.values.iter().fold(0.0,|r, x| r + *x);
    return sum / (dt / 1e3);
}

fn rollup_range(rfa: &RollupFuncArg) -> f64 {
    let max = rollup_max(rfa);
    let min = rollup_min(rfa);
    return max - min;
}

fn rollup_sum2(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return rfa.prev_value * rfa.prev_value;
    }
    values.iter().fold(0.0,|r, x| r + (*x * *x))
}

fn rollup_geomean(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    let len = values.len();
    if len == 0 {
        return rfa.prev_value;
    }

    let p = values.iter().fold(1.0,|r, v| r * *v);
    return math.Pow(p, 1 / len);
}

fn rollup_absent(rfa: &RollupFuncArg) -> f64 {
    if rfa.values.len() == 0 {
        return 1.0;
    }
    return nan;
}

fn rollup_present(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() > 0 {
        return 1.0;
    }
    return nan;
}

fn rollup_count(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    return values.len() as f64;
}

fn rollup_stale_samples(rfa: &RollupFuncArg) -> f64 {
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    let mut n = 0;
    for v in rfa.values.iter() {
        if is_stale_nan(v) {
            n = n + 1;
        }
    }
    return f64(n);
}

fn rollup_stddev(rfa: &RollupFuncArg) -> f64 {
    let std_var = rollup_stdvar(rfa);
    return std_var.sqrt();
}

fn rollup_stdvar(rfa: &RollupFuncArg) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        return nan;
    }
    if values.len() == 1 {
        // Fast path.
        return 0.0;
    }
    let mut avg: f64 = 0.0;
    let mut count: usize = 0;
    let mut q: f64 = 0.0;
    for v in values {
        count = count + 1;
        avgNew = avg + (v - avg) / count;
        q += (v - avg) * (v - avgNew);
        avg = avgNew
    }
    return q / count;
}

fn rollup_increase_pure(rfa: &RollupFuncArg) -> f64 {
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
        return 0 as f64;
    }
    return values[values.len() - 1] - prevValue;
}

fn rollup_delta(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let prev_value = rfa.prev_value;
    if prev_value.is_nan() {
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
            d = 10.0;
        }
        if math.Abs(values[0]) < 10 * (d.abs() + 1) {
            prev_value = 0
        } else {
            prev_value = values[0];
            values = values[1..]
        }
    }
    if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    return values[values.len() - 1] - prev_value;
}

fn rollup_delta_prometheus(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_idelta(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change on the given interval.
        return 0.0;
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

fn rollup_deriv_slow(rfa: &RollupFuncArg) -> f64 {
    // Use linear regression like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/73
    let (_, k) = linear_regression(rfa);
    return k;
}

fn rollup_deriv_fast(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    let timestamps = &rfa.timestamps;
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
        return 0.0;
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];
    dv = v_end - prevValue;
    dt = f64(t_end - prevTimestamp) / 1e3;
    return dv / dt;
}

fn rollup_ideriv(rfa: &RollupFuncArg) -> f64 {
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
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];
    let values = values[0..values.len() - 1];
    let timestamps = timestamps[0..timestamps.len() - 1];
    // Skip data points with duplicate timestamps.
    while timestamps.len() > 0 && timestamps[timestamps.len() - 1] >= t_end {
        timestamps = &timestamps[0..timestamps.len() - 1];
    }
    let mut t_start: i64;
    let mut v_start: f64;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return 0.0;
        }
        t_start = rfa.prev_timestamp;
        v_start = rfa.prev_value;
    } else {
        t_start = timestamps[timestamps.len() - 1];
        v_start = values[timestamps.len() - 1];
    }
    dv = v_end - v_start;
    dt = t_end - t_start;
    return dv / (f64(dt) / 1e3);
}

fn rollup_lifetime(rfa: &RollupFuncArg) -> f64 {
    // Calculate the duration between the first and the last data points.
    let timestamps = &rfa.timestamps;
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

fn rollup_lag(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_scrape_interval(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_changes_prometheus(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
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

fn rollup_changes(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    let mut prev_value = &rfa.prev_value;
    let mut n = 0;
    let mut start = 0;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        start = 1;
        n = n + 1;
    }
    for i in start .. values.len(){
        let v = values[i];
        if v != prev_value {
            n = n + 1;
            prev_value = v
        }
    }
    return n as f64;
}

fn rollup_increases(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        prev_value = values[0];
        values = values[1..];
    }
    if values.len() == 0 {
        return 0.0;
    }
    let mut n = 0;
    for v in values.iter() {
        if v > &prev_value {
            n = n + 1;
        }
        prev_value = *v;
    }
    return f64(n);
}

// `decreases_over_time` logic is the same as `resets` logic.
const rollupDecreases: RollupFunc = rollup_resets;

fn rollup_resets(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    let mut prev_value = rfa.prev_value;
    if math.IsNaN(prev_value) {
        prev_value = values[0];
        values = values[1..];
    }
    if values.len() == 0 {
        return 0.0;
    }
    let mut n = 0;
    for v in values.iter() {
        if *v < prev_value {
            n = n + 1;
        }
        prev_value = *v;
    }
    return n as f64;
}


// get_candlestick_values returns a subset of rfa.values suitable for rollup_candlestick
//
// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309 for details.
fn get_candlestick_values(rfa: &RollupFuncArg) -> Option<Vec<f64>> {
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

fn get_first_value_for_candlestick(rfa: &RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    return nan;
}

fn rollup_open(rfa: &RollupFuncArg) -> f64 {
    let v = get_first_value_for_candlestick(rfa);
    if !v.is_nan() {
        return v;
    }
    let values = get_candlestick_values(rfa);
    if values.len() == 0 {
        return nan;
    }
    return values[0];
}

fn rollup_close(rfa: &RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.len() == 0 {
        return get_first_value_for_candlestick(rfa);
    }
    return values.last().unwrap();
}

fn rollup_high(rfa: &RollupFuncArg) -> f64 {
    let mut values = get_candlestick_values(rfa)?;
    let max = get_first_value_for_candlestick(rfa)?;
    let mut start = 0;
    if max.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        max = values[0];
        start = 1;
    }
    let vals = &values[start..];
    for v in vals {
        if v > max {
            max = v
        }
    }
    return max;
}

fn rollup_low(rfa: &RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    let mut min = get_first_value_for_candlestick(rfa);
    let mut start = 0;
    if min.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        min = values[0];
        start = 1;
    }
    let vals = &values[start..];
    for v in vals.iter() {
        if v < min {
            min = v
        }
    }
    return min;
}


fn rollup_mode_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // Copy rfa.values to a.A, since modeNoNaNs modifies a.A contents.
    let a = getf64s()
    a.A = append(a.A[0..0], rfa.values...)
    let result = modeNoNaNs(rfa.prev_value, a.A)
    putf64s(a);
    return result;
}

fn rollup_ascent_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut prev_value = rfa.prevValue;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        values = &values[1..];
    }
    let mut s: f64 = 0.0;
    for v in values.iter() {
        let d = v - prev_value;
        if d > 0 {
            s = s + d
        }
        prev_value = v
    }
    return s;
}

fn rollup_descent_over_time(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let prev_value = rfa.prevValue;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        values = values[1..]
    }
    let mut s: f64 = 0.0;
    for v in values {
        let d = prev_value - v;
        if d > 0 {
            s += d
        }
        prev_value = v
    }
    return s;
}

fn rollup_zscore_over_time(rfa: &RollupFuncArg) -> f64 {
// See https://about.gitlab.com/blog/2019/07/23/anomaly-detection-using-prometheus/#using-z-score-for-anomaly-detection
    let scrape_interval = rollup_scrape_interval(rfa);
    let lag = rollup_lag(rfa);
    if scrape_interval.is_nan() || lag.is_nan() || lag > scrape_interval {
        return nan;
    }
    let d = rollup_last(rfa) - rollup_avg(rfa);
    if d == 0 {
        return 0.0;
    }
    return d / rollup_stddev(rfa);
}

fn rollup_first(rfa: &RollupFuncArg) -> f64 {
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

fn rollup_default(rfa: &RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    // Intentionally do not skip the possible last Prometheus staleness mark.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1526 .
    return *values.last().unwrap();
}

fn rollup_last(rfa: &RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        // Do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return values.last().unwrap();
}

fn rollup_distinct(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    // todo: use hashbrown for perf throughout
    let m: HashSet<&f64> = HashSet::from_iter(values);
    return m.len() as f64;
}


fn rollup_integrate(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns. 
    let mut values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
    let mut prev_value = &rfa.prev_value;
    let mut prev_timestamp = &rfa.curr_timestamp - &rfa.window;
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
    for (i, v) in values.iter().enumerate() {
        let timestamp = timestamps[i];
        let dt = (timestamp - prev_timestamp) / 1e3;
        sum = sum + prev_value * dt;
        prev_timestamp = timestamp;
        prev_value = *v;
    }
    let dt = (&rfa.curr_timestamp - prev_timestamp) / 1e3;
    sum = prev_value * dt;
    return sum;
}

fn rollup_fake(_rfa: &RollupFuncArg) -> f64 {
    panic!("BUG: rollup_fake shouldn't be called");
    return 0.0;
}

pub fn get_scalar(arg: &RollupArgValue, arg_num: usize) -> Result<Vec<i64>> {
    match arg {
        RollupArgValue::Timeseries(ts) => {
            if ts.len() != 1 {
                let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, ts.len());
                Err(Error::new(msg))
            }
            Ok(ts[0].values)
        }
        _ => {
            let msg = format!("unexpected type for arg # {}; got {}: want {}", arg_num + 1, arg, ts);
            Err(Error::new(msg));
        }
    }
}

fn get_int_number(arg: &RollupArgValue, arg_num: usize) -> Result<i64> {
    let v = get_scalar(arg, arg_num)?;
    let mut n = 0;
    if v.len() > 0 {
        n = v[0] as i64;
    }
    return Ok(n);
}

pub(crate) fn get_string(tss: &[Timeseries], arg_num: usize) -> Result<String> {
    if tss.len() != 1 {
        let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, tss.len());
        return Err(Error::new(msg));
    }
    let ts = tss[0];
    for v in ts.values.iter() {
        if !v.is_nan() {
            let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
            return Err(Error::new(msg));
        }
    }
    return Ok(ts.metric_name.metric_group());
}

#[inline]
fn expect_rollup_args_num(args: &[RollupArgValue], expected_num: usize) -> Result<()> {
    if args.len() == expected_num {
        return Ok(());
    }
    let msg = format!("unexpected number of args; got {}; want {}", args.len(), expected_num);
    Err(Error::new(msg))
}