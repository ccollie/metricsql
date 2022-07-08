use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use lockfree_object_pool::{LinearObjectPool, LinearReusable};
use once_cell::sync::Lazy;
use phf::{phf_map, phf_set};

use lib::{get_float64s, is_stale_nan};
use metricsql::ast::{Expression, ExpressionNode};
use crate::eval::{validate_max_points_per_timeseries};
use crate::{get_timeseries, get_timestamps};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::Timeseries;
use super::timeseries_map::TimeseriesMap;
use crate::functions::aggr::{mode_no_nans, quantile, quantiles};

pub(crate) type RollupArgValue = Vec<Timeseries>;


const nan: f64 = f64::NAN;
const inf: f64 = f64::INFINITY;

/// The maximum interval without previous rows.
pub const MAX_SILENCE_INTERVAL: i64 = 5 * 60 * 1000;

/// The minimum interval for staleness calculations. This could be useful for removing gaps on graphs
/// generated from time series with irregular intervals between samples.
/// Todo: place as field on EvalConfig
pub const MIN_STALENESS_INTERVAL: i64 = 0;

/// The maximum interval for staleness calculations. By default it is automatically calculated from
/// the median interval between samples. This flag could be useful for tuning Prometheus data model
/// closer to Influx-style data model.
/// See https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness for details.
/// See also '-search.setLookbackToStep' flag
/// Todo: place as field on EvalConfig
pub const MAX_STALENESS_INTERVAL: i64 = 0;

pub(crate) type NewRollupFunc = fn(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc>;

pub(crate) static NEW_ROLLUP_FUNC_FAKE: Lazy<NewRollupFunc> = Lazy::new(|| new_rollup_func_one_arg(rollup_fake));
pub(crate) static NEW_DEFAULT_ROLLUP_FUNC: Lazy<NewRollupFunc> = Lazy::new(|| new_rollup_func_one_arg(rollup_default));

static ROLLUP_FUNCTIONS: phf::Map<&'static str, NewRollupFunc> = phf_map! {
	"absent_over_time" =>         new_rollup_func_one_arg(rollup_absent),
	"aggr_over_time" =>           new_rollup_func_two_args(rollup_fake),
	"ascent_over_time" =>         new_rollup_func_one_arg(rollup_ascent_over_time),
	"avg_over_time" =>            new_rollup_func_one_arg(rollup_avg),
	"changes" =>                  new_rollup_func_one_arg(rollup_changes),
	"changes_prometheus" =>       new_rollup_func_one_arg(rollup_changes_prometheus),
	"count_eq_over_time" =>       new_rollup_count_eq,
	"count_gt_over_time" =>       new_rollup_count_gt,
	"count_le_over_time" =>       new_rollup_count_le,
	"count_ne_over_time" =>       new_rollup_count_ne,
	"count_over_time" =>          new_rollup_func_one_arg(rollup_count),
	"decreases_over_time" =>      new_rollup_func_one_arg(rollup_decreases),
	"default_rollup" =>           new_rollup_func_one_arg(rollup_default), // default rollup func
	"delta" =>                    new_rollup_func_one_arg(rollup_delta),
	"delta_prometheus" =>         new_rollup_func_one_arg(rollup_delta_prometheus),
	"deriv" =>                    new_rollup_func_one_arg(rollup_deriv_slow),
	"deriv_fast" =>               new_rollup_func_one_arg(rollup_deriv_fast),
	"descent_over_time" =>        new_rollup_func_one_arg(rollup_descent_over_time),
	"distinct_over_time" =>       new_rollup_func_one_arg(rollup_distinct),
	"duration_over_time" =>       new_rollup_duration_over_time,
	"first_over_time" =>          new_rollup_func_one_arg(rollup_first),
	"geomean_over_time" =>        new_rollup_func_one_arg(rollup_geomean),
	"histogram_over_time" =>      new_rollup_func_one_arg(rollup_histogram),
	"hoeffding_bound_lower" =>    new_rollup_hoeffding_bound_lower,
	"hoeffding_bound_upper" =>    new_rollup_hoeffding_bound_upper,
	"holt_winters" =>             new_rollup_holt_winters,
	"idelta" =>                   new_rollup_func_one_arg(rollup_idelta),
	"ideriv" =>                   new_rollup_func_one_arg(rollup_ideriv),
	"increase" =>                 new_rollup_func_one_arg(rollup_delta),           // + rollupFuncsRemoveCounterResets
	"increase_prometheus" =>      new_rollup_func_one_arg(rollup_delta_prometheus), // + rollupFuncsRemoveCounterResets
	"increase_pure" =>            new_rollup_func_one_arg(rollup_increase_pure),    // + rollupFuncsRemoveCounterResets
	"increases_over_time" =>      new_rollup_func_one_arg(rollup_increases),
	"integrate" =>                new_rollup_func_one_arg(rollup_integrate),
	"irate" =>                    new_rollup_func_one_arg(rollup_ideriv), // + rollupFuncsRemoveCounterResets
	"lag" =>                      new_rollup_func_one_arg(rollup_lag),
	"last_over_time" =>           new_rollup_func_one_arg(rollup_last),
	"lifetime" =>                 new_rollup_func_one_arg(rollup_lifetime),
	"max_over_time" =>            new_rollup_func_one_arg(rollup_max),
	"min_over_time" =>            new_rollup_func_one_arg(rollup_min),
	"mode_over_time" =>           new_rollup_func_one_arg(rollup_mode_over_time),
	"predict_linear" =>           new_rollup_predict_linear,
	"present_over_time" =>        new_rollup_func_one_arg(rollup_present),
	"quantile_over_time" =>       new_rollup_quantile,
	"quantiles_over_time" =>      new_rollup_quantiles,
	"range_over_time" =>          new_rollup_func_one_arg(rollup_range),
	"rate" =>                     new_rollup_func_one_arg(rollup_deriv_fast), // + rollupFuncsRemoveCounterResets
	"rate_over_sum" =>            new_rollup_func_one_arg(rollup_rate_over_sum),
	"resets" =>                   new_rollup_func_one_arg(rollup_resets),
	"rollup" =>                   *NEW_ROLLUP_FUNC_FAKE,
	"rollup_candlestick" =>       *NEW_ROLLUP_FUNC_FAKE,
	"rollup_delta" =>             *NEW_ROLLUP_FUNC_FAKE,
	"rollup_deriv" =>             *NEW_ROLLUP_FUNC_FAKE,
	"rollup_increase" =>          *NEW_ROLLUP_FUNC_FAKE, // + rollupFuncsRemoveCounterResets
	"rollup_rate" =>              *NEW_ROLLUP_FUNC_FAKE, // + rollupFuncsRemoveCounterResets
	"rollup_scrape_interval" =>   *NEW_ROLLUP_FUNC_FAKE,
	"scrape_interval" =>          new_rollup_func_one_arg(rollup_scrape_interval),
	"share_gt_over_time" =>       new_rollup_share_gt,
	"share_le_over_time" =>       new_rollup_share_le,
	"stale_samples_over_time" =>  new_rollup_func_one_arg(rollup_stale_samples),
	"stddev_over_time" =>         new_rollup_func_one_arg(rollup_stddev),
	"stdvar_over_time" =>         new_rollup_func_one_arg(rollup_stdvar),
	"sum_over_time" =>            new_rollup_func_one_arg(rollup_sum),
	"sum2_over_time" =>           new_rollup_func_one_arg(rollup_sum2),
	"tfirst_over_time" =>         new_rollup_func_one_arg(rollup_tfirst),
	// `timestamp` function must return timestamp for the last datapoint on the current window
	// in order to properly handle offset and timestamps unaligned to the current step.
	// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415 for details.
	"timestamp" =>               new_rollup_func_one_arg(rollup_tlast),
	"timestamp_with_name" =>     new_rollup_func_one_arg(rollup_tlast), // + rollupFuncsKeepMetricName
	"tlast_change_over_time" =>  new_rollup_func_one_arg(rollup_tlast_change),
	"tlast_over_time" =>         new_rollup_func_one_arg(rollup_tlast),
	"tmax_over_time" =>          new_rollup_func_one_arg(rollup_tmax),
	"tmin_over_time" =>          new_rollup_func_one_arg(rollup_tmin),
	"zscore_over_time" =>        new_rollup_func_one_arg(rollup_zscore_over_time),
};


#[derive(Default, Clone)]
pub(crate) struct RollupFuncArg {
    /// The value preceding values if it fits staleness interval.
    prev_value: f64,

    /// The timestamp for prev_value.
    prev_timestamp: i64,

    /// Values that fit window ending at curr_timestamp.
    values: Vec<f64>,

    /// Timestamps for values.
    timestamps: Vec<i64>,

    /// Real value preceding values without restrictions on staleness interval.
    real_prev_value: f64,

    /// Real value which goes after values.
    real_next_value: f64,

    /// Current timestamp for rollup evaluation.
    curr_timestamp: i64,

    /// Index for the currently evaluated point relative to time range for query evaluation.
    idx: usize,

    /// Time window for rollup calculations.
    window: i64,

    tsm: Option<TimeseriesMap>,
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
        if let Some(mut tsm) = self.tsm {
            tsm.reset()
        }
    }
}

/// RollupFunc must return rollup value for the given rfa.
///
/// prev_value may be nan, values and timestamps may be empty.
pub(crate) type RollupFunc = fn(rfa: &mut RollupFuncArg) -> f64;

trait RollupFn: Fn(&RollupFuncArg) -> f64 {}

/// implement `Rollup` on any type that implements `Fn(&RollupFuncArg) -> f64`.
impl<T> RollupFn for T where T: Fn(&RollupFuncArg) -> f64 {}

static ROLLUP_AGGR_FUNCTIONS: phf::Map<&'static str, RollupFunc> = phf_map! {
    "absent_over_time" =>        rollup_absent,
	"ascent_over_time" =>        rollup_ascent_over_time,
	"avg_over_time" =>           rollup_avg,
	"changes" =>                 rollup_changes,
	"count_over_time" =>         rollup_count,
	"decreases_over_time" =>     rollup_decreases,
	"default_rollup" =>          rollup_default,
	"delta" =>                   rollup_delta,
	"deriv" =>                   rollup_deriv_slow,
	"deriv_fast" =>              rollup_deriv_fast,
	"descent_over_time" =>       rollup_descent_over_time,
	"distinct_over_time" =>      rollup_distinct,
	"first_over_time" =>         rollup_first,
	"geomean_over_time" =>       rollup_geomean,
	"idelta" =>                  rollup_idelta,
	"ideriv" =>                  rollup_ideriv,
	"increase" =>                rollup_delta,
	"increase_pure" =>           rollup_increase_pure,
	"increases_over_time" =>     rollup_increases,
	"integrate" =>               rollup_integrate,
	"irate" =>                   rollup_ideriv,
	"lag" =>                     rollup_lag,
	"last_over_time" =>          rollup_last,
	"lifetime" =>                rollup_lifetime,
	"max_over_time" =>           rollup_max,
	"min_over_time" =>           rollup_min,
	"mode_over_time" =>          rollup_mode_over_time,
	"present_over_time" =>       rollup_present,
	"range_over_time" =>         rollup_range,
	"rate" =>                    rollup_deriv_fast,
	"rate_over_sum" =>           rollup_rate_over_sum,
	"resets" =>                  rollup_resets,
	"scrape_interval" =>         rollup_scrape_interval,
	"stale_samples_over_time" => rollup_stale_samples,
	"stddev_over_time" =>        rollup_stddev,
	"stdvar_over_time" =>        rollup_stdvar,
	"sum_over_time" =>           rollup_sum,
	"sum2_over_time" =>          rollup_sum2,
	"tfirst_over_time" =>        rollup_tfirst,
	"timestamp" =>               rollup_tlast,
	"timestamp_with_name" =>     rollup_tlast,
	"tlast_change_over_time" =>  rollup_tlast_change,
	"tlast_over_time" =>         rollup_tlast,
	"tmax_over_time" =>          rollup_tmax,
	"tmin_over_time" =>          rollup_tmin,
	"zscore_over_time" =>        rollup_zscore_over_time,
};

/// We can increase lookbehind window in square brackets for these functions
/// if the given window doesn't contain enough samples for calculations.
///
/// This is needed in order to return the expected non-empty graphs when zooming in the graph in Grafana,
/// which is built with `func_name(metric[$__interval])` query.
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

/// These functions don't change physical meaning of input time series,
/// so they don't drop metric name
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

pub(crate) fn rollup_func_keeps_metric_name(name: &str) -> bool {
    let lower = name.to_lowercase().as_str();
    ROLLUP_FUNCTIONS_KEEP_METRIC_NAME.contains(lower)
}

pub(crate) fn get_rollup_func(func_name: &str) -> Option<&NewRollupFunc> {
    let lower = func_name.to_lowercase().as_str();
    return ROLLUP_FUNCTIONS.get(lower);
}

// todo: use in optimize so its cached in the ast
pub(crate) fn get_rollup_aggr_func_names(expr: &Expression) -> RuntimeResult<Vec<String>> {
    let fe = match expr {
        Expression::Aggregation(afe) => {
            // This is for incremental aggregate function case:
            //
            //     sum(aggr_over_time(...))
            // See aggr_incremental.rs for details.
            let _expr = &afe.args[0];
            match _expr.deref() {
                Expression::Function(f) => Some(f),
                _ => None
            }
        },
        Expression::Function(fe) => Some(fe),
        _ => None
    };

    return match fe {
        None => {
            let msg = format!("BUG: unexpected expression; want metricsql.FuncExpr; got {}; value: {}", expr.kind(), expr);
            Err(RuntimeError::from(msg))
        },
        Some(fe_) => {
            if fe_.name != "aggr_over_time" {
                let msg = format!("BUG: unexpected function name: {}; want `aggr_over_time`", fe_.name);
                return Err(RuntimeError::from(msg));
            }

            let arg_len = fe_.args.len();
            if arg_len != 2 {
                let msg = format!("unexpected number of args to aggr_over_time(); got {}; want {}", arg_len, 2);
                return Err(RuntimeError::from(msg));
            }

            let arg = &fe_.args[0].as_ref();
            let mut aggr_func_names: Vec<String> = Vec::with_capacity(1);
            match arg {
                Expression::String(se) => {
                    aggr_func_names.push(se.s.to_string());
                },
                Expression::Function(fe) => {
                    if fe.name.len() > 0 {
                        let msg = format!("{} cannot be passed to aggr_over_time(); expecting quoted aggregate function name",
                                          arg);
                        return Err(RuntimeError::General(msg));
                    }
                    for exp in fe.args {
                        match *exp {
                            Expression::String(se) => {
                                let name = &se.s;
                                if ROLLUP_AGGR_FUNCTIONS.get(&name).is_some() {
                                    aggr_func_names.push(se.s);
                                } else {
                                    let msg = format!("{} cannot be used in `aggr_over_time` function; expecting quoted aggregate function name", name);
                                    return Err(RuntimeError::General(msg));
                                }
                            },
                            _ => {
                                let msg = format!("{} cannot be passed here; expecting quoted aggregate function name", exp);
                                return Err(RuntimeError::General(msg));
                            }
                        }
                    }
                },
                _ => {
                    let msg = format!("{} cannot be passed to aggr_over_time(); expecting a list of quoted aggregate function names",
                                      arg);
                    return Err(RuntimeError::General(msg));
                }
            }

            Ok(aggr_func_names)
        }
    }

}

pub(crate) type PreFunc = fn(values: &mut [f64], timestamps: &[i64]) -> ();

pub(crate) fn get_rollup_configs(
    name: &str,
    rf: &RollupFunc,
    expr: &Expression,
    start: i64, end: i64, step: i64, window: i64,
    max_points_per_series: usize,
    lookback_delta: i64,
    shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<(PreFunc, Vec<RollupConfig>)> {

    let mut pre_func: PreFunc = |values, timestamps| {};

    if ROLLUP_FUNCTIONS_REMOVE_COUNTER_RESETS.contains(name) {
        pre_func = |values, timestamps| {
            remove_counter_resets(values);
        }
    }

    let may_adjust_window = ROLLUP_FUNCTIONS_CAN_ADJUST_WINDOW.contains(name);
    let is_default_rollup = name == "default_rollup";

    let new_rollup_config = |rf: RollupFunc, tag_value: &str| -> RollupConfig {
        // todo: get from object pool
        return RollupConfig {
            tag_value: tag_value.to_string(),
            func: rf,
            start,
            end,
            step,
            window,
            may_adjust_window,
            lookback_delta,
            timestamps: shared_timestamps.clone(),
            is_default_rollup,
            max_points_per_timeseries: max_points_per_series
        }
    };
    
    let append_rollup_configs = |mut dst: &Vec<RollupConfig>| {
        dst.push(new_rollup_config(rollup_min, "min"));
        dst.push(new_rollup_config(rollup_max, "max"));
        dst.push(new_rollup_config(rollup_avg, "avg"));
    };

    // todo: tinyvec
    let mut rcs: Vec<RollupConfig> = Vec::with_capacity(1);
    match name {
        "rollup" => append_rollup_configs(&rcs),
        "rollup_rate" | "rollup_deriv" => {
            let mut pre_func_prev = pre_func;
            pre_func = |values, timestamps| {
                pre_func_prev(values, timestamps);
                deriv_values(values, timestamps)
            };
            append_rollup_configs(&mut rcs);
        }
        "rollup_increase" | "rollup_delta" => {
            let mut pre_func_prev = pre_func;
            pre_func = |values, timestamps| {
                pre_func_prev(values, timestamps);
                delta_values(values)
            };
            append_rollup_configs(&rcs);
        }
        "rollup_candlestick" => {
            rcs.push(new_rollup_config(rollup_open, "open"));
            rcs.push(new_rollup_config(rollup_close, "close"));
            rcs.push(new_rollup_config(rollup_low, "low"));
            rcs.push(new_rollup_config(rollup_high, "high"));
        },
        "rollup_scrape_interval" => {
            let pre_func_prev = pre_func;
            pre_func = |values, timestamps| {
                pre_func_prev(values, timestamps);
                // Calculate intervals in seconds between samples.
                let mut ts_secs_prev = nan;
                for (i, ts) in timestamps.iter().enumerate() {
                    let ts_secs = (ts / 1000) as f64;
                    values[i] = ts_secs - ts_secs_prev;
                    ts_secs_prev = ts_secs;
                };
                if values.len() > 1 {
                    // Overwrite the first NaN interval with the second interval,
                    // So min, max and avg rollups could be calculated properly,
                    // since they don't expect to receive NaNs.
                    values[0] = values[1]
                }
            };
            append_rollup_configs(&rcs);
        }
        "aggr_over_time" => {
            match get_rollup_aggr_func_names(expr) {
                Err(err) => {
                    return Err(RuntimeError::ArgumentError(format!("invalid args to {}", expr)))
                },
                Ok(func_names) => {
                    for aggr_func_name in func_names {
                        if ROLLUP_FUNCTIONS_REMOVE_COUNTER_RESETS.contains(&aggr_func_name) {
                            // There is no need to save the previous pre_func, since it is either empty or the same.
                            pre_func = |values, timestamps| {
                                remove_counter_resets(values)
                            }
                        }
                        let rf = ROLLUP_AGGR_FUNCTIONS.get(&aggr_func_name).unwrap();
                        rcs.push(new_rollup_config(*rf, &aggr_func_name));
                    }
                }
            }
        },
        _ => {
            rcs.push(new_rollup_config(*rf, ""));
        }
    }
    Ok((pre_func, rcs))
}

#[derive(Debug, Clone)]
pub(crate) struct RollupConfig {
    /// This tag value must be added to "rollup" tag if non-empty.
    pub tag_value: String,
    func: RollupFunc,
    start: i64,
    end: i64,
    step: i64,
    window: i64,

    /// Whether window may be adjusted to 2 x interval between data points.
    /// This is needed for functions which have dt in the denominator
    /// such as rate, deriv, etc.
    /// Without the adjustment their value would jump in unexpected directions
    /// when using window smaller than 2 x scrape_interval.
    may_adjust_window: bool,

    timestamps: Arc<Vec<i64>>,

    /// lookback_delta is the analog to `-query.lookback-delta` from Prometheus world.
    lookback_delta: i64,

    /// Whether default_rollup is used.
    is_default_rollup: bool,

    /// The maximum number of points which can be generated per each series.
    max_points_per_timeseries: usize
}

impl RollupConfig {
    fn clone_with_fn(&self, rollupFn: &RollupFunc, tag_value: &str) -> Self {
        return RollupConfig {
            tag_value: tag_value.to_string(),
            func: *rollupFn,
            start: self.start,
            end: self.end,
            step: self.step,
            window: self.window,
            may_adjust_window: self.may_adjust_window,
            lookback_delta: self.lookback_delta,
            timestamps: self.timestamps.clone(),
            is_default_rollup: self.is_default_rollup,
            max_points_per_timeseries: self.max_points_per_timeseries
        }
    }

    /// calculates rollups for the given timestamps and values, appends
    /// them to dst_values and returns results.
    ///
    /// rc.timestamps are used as timestamps for dst_values.
    ///
    /// timestamps must cover time range [rc.start - rc.Window - MAX_SILENCE_INTERVAL ... rc.end].
    ///
    /// do cannot be called from concurrent goroutines.
    pub(crate) fn exec(&mut self, dst_values: &mut Vec<f64>, values: &[f64], timestamps: &[i64]) -> RuntimeResult<()> {
        self.do_internal(dst_values, None, values, timestamps)
    }

    /// calculates rollups for the given timestamps and values and puts them to tsm.
    pub(crate) fn do_timeseries_map(&mut self, tsm: &mut TimeseriesMap, values: &[f64], timestamps: &[i64]) {
        let mut ts = get_timeseries();
        self.do_internal(&mut ts.values,Some(tsm), values, timestamps)?;
    }

    pub(super) fn get_timestamps(&self) -> RuntimeResult<Vec<i64>> {
        get_timestamps(self.start, self.end, self.step,
                       self.max_points_per_timeseries as usize)
    }

    fn do_internal(&mut self,
                   dst_values: &mut Vec<f64>,
                   tsm: Option<&mut TimeseriesMap>,
                   values: &[f64],
                   timestamps: &[i64]) -> RuntimeResult<()> {

        // Sanity checks.
        self.validate()?;

        // Extend dst_values in order to remove mallocs below.
        dst_values.reserve(self.timestamps.len());

        let scrape_interval = get_scrape_interval(&timestamps);
        let mut max_prev_interval = get_max_prev_interval(scrape_interval);
        if self.lookback_delta > 0 && max_prev_interval > self.lookback_delta {
            max_prev_interval = self.lookback_delta
        }
        if MIN_STALENESS_INTERVAL > 0 {
            let msi = MIN_STALENESS_INTERVAL;
            if msi > 0 && max_prev_interval < msi {
                max_prev_interval = msi
            }
        }
        let mut window = self.window;
        if window <= 0 {
            window = self.step;
            if self.is_default_rollup && self.lookback_delta > 0 && window > self.lookback_delta {
                // Implicit window exceeds -search.maxStalenessInterval, so limit it to -search.maxStalenessInterval
                // according to https://github.com/VictoriaMetrics/VictoriaMetrics/issues/784
                window = self.lookback_delta
            }
        }
        if self.may_adjust_window && window < max_prev_interval {
            window = max_prev_interval
        }
        // todo: just init on stack
        let mut rfa = get_rollup_func_arg();
        rfa.idx = 0;
        rfa.window = window;
        rfa.tsm = tsm;

        let mut i = 0;
        let mut j = 0;
        let mut ni = 0;
        let mut nj = 0;

        for tEnd in self.timestamps.iter() {
            let t_start = *tEnd - window;
            let ni = seek_first_timestamp_idx_after(&timestamps[i..], t_start, ni);
            i += ni;
            if j < i {
                j = i;
            }
            let nj = seek_first_timestamp_idx_after(&timestamps[j..], *tEnd, nj);
            j += nj;

            rfa.prev_value = nan;
            rfa.prev_timestamp = t_start - max_prev_interval;
            if i < timestamps.len() && i > 0 && timestamps[i-1] > rfa.prev_timestamp {
                rfa.prev_value = values[i - 1];
                rfa.prev_timestamp = timestamps[i-1];
            }

            rfa.values.clear();
            rfa.timestamps.clear();
            rfa.values.extend_from_slice(&values[i..j]);
            rfa.timestamps.extend_from_slice(&timestamps[i..j]);

            if i > 0 {
                rfa.real_prev_value = values[i-1];
            } else {
                rfa.real_prev_value = nan;
            }
            if j < values.len() {
                rfa.real_next_value = values[j];
            } else {
                rfa.real_next_value = nan;
            }
            rfa.curr_timestamp = *tEnd;
            let value = (self.func)(&rfa);
            rfa.idx += 1;
            dst_values.push(value);
        }

        Ok(())
    }

    fn validate(self) -> RuntimeResult<()> {
        // Sanity checks.
        if self.step <= 0 {
            let msg = format!("BUG: Step must be bigger than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        if self.start > self.end {
            let msg = format!("BUG: start cannot exceed End; got {} vs {}", self.start, self.end);
            return Err(RuntimeError::from(msg));
        }
        if self.window < 0 {
            let msg = format!("BUG: Window must be non-negative; got {}", self.window);
            return Err(RuntimeError::from(msg));
        }
        match validate_max_points_per_timeseries(self.start, self.end, self.step, self.max_points_per_timeseries) {
            Err(err) => {
                let msg = format!("BUG: {:?}; this must be validated before the call to rollupConfig.exec", err);
                return Err(RuntimeError::from(msg))
            },
            _ => Ok(())
        }
    }
}


fn new_rollup_func_one_arg(rf: RollupFunc) -> NewRollupFunc {
    |args: &[RollupArgValue]| -> RuntimeResult<&RollupFunc> {
        expect_rollup_args_num(args, 1)?;
        Ok(rf)
    }
}

fn new_rollup_func_two_args(rf: &RollupFunc) -> fn(&[RollupArgValue]) -> RuntimeResult<&RollupFunc> {
    |args: &[RollupArgValue]| -> RuntimeResult<&RollupFunc> {
        expect_rollup_args_num(&args, 2)?;
        Ok(rf)
    }
}

fn seek_first_timestamp_idx_after(timestamps: &[i64], seek_timestamp: i64, n_hint: usize) -> usize {
    let mut ts = timestamps;
    let count = timestamps.len();

    if count == 0 || timestamps[0] > seek_timestamp {
        return 0;
    }
    let mut start_idx = n_hint - 2;
    start_idx = start_idx.clamp(0, count - 1);

    let mut end_idx = n_hint + 2;
    if end_idx > count {
        end_idx = count
    }
    if start_idx > 0 && timestamps[start_idx] <= seek_timestamp {
        ts = &timestamps[start_idx..];
        end_idx -= start_idx
    } else {
        start_idx = 0
    }
    if end_idx < count && timestamps[end_idx] > seek_timestamp {
        ts = &timestamps[0..end_idx];
    }
    if count < 16 {
        // Fast path: the number of timestamps to search is small, so scan them all.
        for (i, timestamp) in ts.iter().enumerate() {
            if *timestamp > seek_timestamp {
                return start_idx + i;
            }
        }
        return start_idx + ts.len();
    }
    // Slow path: too big timestamps.len(), so use binary search.
    let requested = seek_timestamp + 1;
    match ts.binary_search(&requested) {
        Ok(pos) => start_idx + pos,
        Err(suggested) => start_idx + suggested
    }
}

fn get_scrape_interval(timestamps: &[i64]) -> i64 {
    if timestamps.len() < 2 {
        return MAX_SILENCE_INTERVAL;
    }

    // Estimate scrape interval as 0.6 quantile for the first 20 intervals.
    let mut ts_prev = timestamps[0];
    let mut timestamps = &timestamps[1..];
    let len = timestamps.len().clamp(0, 20);

    let mut intervals = tiny_vec!([f64; 20]);
    for i in 0 .. len {
        let ts = timestamps[i];
        intervals.push((ts - ts_prev) as f64);
        ts_prev = ts
    }
    let scrape_interval = quantile(0.6, &intervals) as i64;
    if scrape_interval <= 0 {
        return MAX_SILENCE_INTERVAL;
    }
    return scrape_interval;
}

fn get_max_prev_interval(scrape_interval: i64) -> i64 {
    // Increase scrape_interval more for smaller scrape intervals in order to hide possible gaps
    // when high jitter is present.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/139 .
    if scrape_interval <= 2_000i64 {
        return scrape_interval + 4 * scrape_interval;
    }
    if scrape_interval <= 4_000i64 {
        return scrape_interval + 2 * scrape_interval;
    }
    if scrape_interval <= 8_000i64 {
        return scrape_interval + scrape_interval;
    }
    if scrape_interval <= 16_000i64 {
        return scrape_interval + scrape_interval / 2;
    }
    if scrape_interval <= 32_000i64 {
        return scrape_interval + scrape_interval / 4;
    }
    return scrape_interval + scrape_interval / 8;
}

fn remove_counter_resets(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from vmstorage.
    if values.len() == 0 {
        return;
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];

    for v in values.iter_mut() {
        let d = *v - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely a partial counter reset.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2787
                *v = prev_value;
            } else {
                correction += prev_value;
            }
        }
        prev_value = *v;
        *v += correction;
    }
}

fn delta_values(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
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

fn deriv_values(values: &mut [f64], timestamps: &[i64]) {
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
        let dt = (ts - prev_ts) as f64 / 1e3_f64;
        prev_deriv = (v - prev_value) / dt;
        values[i] = prev_deriv;
        prev_value = *v;
        prev_ts = ts
    }
    values[values.len() - 1] = prev_deriv
}

fn new_rollup_holt_winters(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(args, 3)?;
    let sfs = get_scalar(&args, 1)?;
    let tfs = get_scalar(&args, 2)?;

    let res = move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = &rfa.values;
        if values.len() == 0 {
            return rfa.prev_value;
        }
        let sf = sfs[rfa.idx];
        if sf <= 0.0 || sf >= 1.0 {
            return nan;
        }
        let tf = tfs[rfa.idx];
        if tf <= 0.0 || tf >= 1.0 {
            return nan;
        }

        let mut vals = &values[0..];

        // See https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing .
        // TODO: determine whether this shit really works.
        let mut s0 = rfa.prev_value;
        if s0.is_nan() {
            s0 = vals[0];
            vals = &vals[1..];
            if vals.len() == 0 {
                return s0;
            }
        }
        let mut b0 = vals[0] - s0;
        for v in vals {
            let s1 = sf * v + (1.0 - sf) * (s0 + b0);
            let b1 = tf * (s1 - s0) + (1.0 - tf) * b0;
            s0 = s1;
            b0 = b1
        }
        return s0;
    };

    Ok(res)
}

fn new_rollup_predict_linear(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(&args, 2)?;
    let secs = get_scalar(args, 1)?;

    let f = |rfa: &mut RollupFuncArg| -> f64 {
        let (v, k) = linear_regression(rfa);
        if v.is_nan() {
            return nan;
        }
        return v + k * secs[rfa.idx];
    };
    
    Ok(f)
}

fn linear_regression(rfa: &mut RollupFuncArg) -> (f64, f64) {
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
        let dt = (timestamps[i] - intercept_time) as f64 / 1e3_f64;
        v_sum += v;
        t_sum += dt;
        tv_sum += dt * v;
        tt_sum += dt * dt
    }
    let mut k: f64 = 0.0;
    let n = n as f64;
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
    for v in &values[1..] {
        if *v != v_prev {
            return false;
        }
        v_prev = *v
    }
    return true;
}

fn new_rollup_duration_over_time(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let d_maxs = get_scalar(args, 1)?;

   let f = move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.timestamps.len() == 0 {
            return nan;
        }
        let mut t_prev = rfa.timestamps[0];
        let mut d_sum: i64 = 0;
        let d_max = (d_maxs[rfa.idx] * 1000_f64) as i64;
        for t in rfa.timestamps.iter() {
            let d = t - t_prev;
            if d <= d_max {
                d_sum += d;
            }
            t_prev = *t
        }
        return (d_sum as f64 / 1000_f64) as f64
    };
    
    Ok(f)
}

fn new_rollup_share_le(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    return new_rollup_share_filter(args, count_filter_le);
}

fn count_filter_le(values: &[f64], le: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if *v <= le {
            n += 1;
        }
    }
    n
}

fn new_rollup_share_gt(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    return new_rollup_share_filter(args, count_filter_gt);
}

#[inline]
fn count_filter_gt(values: &[f64], gt: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if *v > gt {
            n += 1;
        }
    }
    return n;
}

#[inline]
fn count_filter_eq(values: &[f64], eq: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if *v == eq {
            n += 1;
        }
    }
    return n;
}

#[inline]
fn count_filter_ne(values: &[f64], ne: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if *v != ne {
            n += 1;
        }
    }
    return n;
}

fn new_rollup_share_filter(args: &Vec<RollupArgValue>, count_filter: fn(values: &[f64], limit: f64) -> i32) -> RuntimeResult<RollupFunc> {
    let rf = new_rollup_count_filter(args, count_filter)?;
    let f = move |rfa: &mut RollupFuncArg| -> f64 {
        let n = rf(&rfa);
        return n / rfa.values.len() as f64;
    };
    Ok(f)
}

fn new_rollup_count_le(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    new_rollup_count_filter(args, count_filter_le)
}

fn new_rollup_count_gt(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    new_rollup_count_filter(args, count_filter_gt)
}

fn new_rollup_count_eq(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_eq);
}

fn new_rollup_count_ne(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    return new_rollup_count_filter(args, count_filter_ne);
}

fn new_rollup_count_filter(args: &Vec<RollupArgValue>, count_filter: fn(values: &[f64], limit: f64) -> i32) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(&args, 2)?;
    let limits = get_scalar(args, 1)?;
    Ok(|rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = &rfa.values;
        if values.len() == 0 {
            return nan;
        }
        let limit = limits[rfa.idx];
        return count_filter(&values, limit as f64) as f64;
    })
}

fn new_rollup_hoeffding_bound_lower(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(args, 0)?;
    let f = move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(&rfa, &phis);
        return avg - bound;
    };

    Ok(f)
}

fn new_rollup_hoeffding_bound_upper(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(args, 0)?;
    let f = move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(&rfa, &phis);
        return avg + bound;
    };

    Ok(f)
}

fn rollup_hoeffding_bound_internal(rfa: &mut RollupFuncArg, phis: &[f64]) -> (f64, f64) {
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

    // let bound = v_range * math.Sqrt(math.Log(1 / (1 - phi)) / (2 * values.len()));
    let bound = v_range * ((1.0 / (1.0 - phi)).ln() / (2 * values.len()) as f64 ).sqrt();
    return (bound, v_avg)
}

fn new_rollup_quantiles(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    if args.len() < 3 {
        let msg = format!("unexpected number of args: {}; want at least 3 args", args.len());
        return Err(RuntimeError::ArgumentError(msg));
    }
    let mut tss_phi: &Vec<Timeseries> = &args[0];

    let phi_label = get_string(tss_phi, 0)?;
    let phi_args = &args[1 .. args.len() - 1];
    let mut phis = Vec::with_capacity(phi_args.len());
    // todo: smallvec ??
    let mut phi_strs: Vec<String> = Vec::with_capacity(phi_args.len());

    for i in 0 .. phi_args.len() {
        match get_scalar(phi_args, i + 1) {
            Ok(values) => {
                phis[i] = values[0];
                phi_strs[i] = format!("{}", values[0]);                
            },
            Err(err) => {
                return Err(RuntimeError::ArgumentError(format!("cannot obtain phi from arg #{}", i + 1)));
            }
        }
    }

    let f = move |rfa: &mut RollupFuncArg| -> f64 {
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
        // tinyvec ?
        let mut qs = get_float64s(phis.len());
        quantiles(qs.deref_mut(), &phis, values);
        let idx = rfa.idx;
        let mut tsm = &rfa.tsm.unwrap();
        for (i, phiStr) in phi_strs.iter().enumerate() {
            let mut ts = tsm.get_or_create_timeseries(&phi_label, phiStr);
            ts.values[idx] = qs[i];
        }

        return nan;
    };

    Ok(f)
}

fn new_rollup_quantile(args: &Vec<RollupArgValue>) -> RuntimeResult<RollupFunc> {
    expect_rollup_args_num(args, 2)?;
    let phis = get_scalar(args, 0)?;
    let rf = |rfa: &mut RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let mut values = &rfa.values;
        let phi = phis[rfa.idx];
        quantile(phi, values)
    };

    Ok(rf)
}

fn rollup_histogram(rfa: &mut RollupFuncArg) -> f64 {
    let mut values = &rfa.values;
    let mut tsm = &rfa.tsm.unwrap();
    tsm.reset();
    for v in values {
        tsm.update(*v);
    }
    let mut idx = rfa.idx;
    for bucket in tsm.non_zero_buckets() {
        let mut ts = tsm.get_or_create_timeseries("vmrange", bucket.vm_range);
        ts.values[idx] = bucket.count as f64;
    }
    return nan;
}

fn rollup_avg(rfa: &mut RollupFuncArg) -> f64 {
    // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
    // since it is slower and has no significant benefits in precision.

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    let sum: f64 = values.iter().fold(0.0,|r, x| r + *x);
    return sum / values.len() as f64;
}

fn rollup_min(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    let mut min_value = values[0];
    for v in values {
        if v < &min_value {
            min_value = *v;
        }
    }
    return min_value;
}

fn rollup_max(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    *values.iter().max().unwrap()
}

fn rollup_tmin(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
    if values.len() == 0 {
        return nan;
    }
    let mut min_value = values[0];
    let mut min_timestamp = timestamps[0];
    for (i, v) in values.iter().enumerate() {
        // Get the last timestamp for the minimum value as most users expect.
        if v <= &min_value {
            min_value = *v;
            min_timestamp = timestamps[i];
        }
    }
    return (min_timestamp as f64 / 1e3_f64) as f64;
}

fn rollup_tmax(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
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
    return (max_timestamp as f64 / 1e3_f64) as f64;
}

fn rollup_tfirst(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return timestamps[0] as f64 / 1e3_f64;
}

fn rollup_tlast(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return timestamps[timestamps.len() - 1] as f64 / 1e3_f64;
}

fn rollup_tlast_change(rfa: &mut RollupFuncArg) -> f64 {
// There is no need in handling NaNs here, since they must be cleaned up
// before calling rollup fns.
    let mut values = &rfa.values[0..];
    if values.len() == 0 {
        return nan;
    }
    let mut timestamps = &rfa.timestamps;
    let last = values.len() - 1;
    let last_value = values[last];
    values = &values[0..last];
    let mut i = last;
    for value in values.iter().rev() {
        if *value != last_value {
            return timestamps[i + 1] as f64 / 1e3_f64;
        }
        i -= 1;
    }
    if rfa.prev_value.is_nan() || rfa.prev_value != last_value {
        return timestamps[0] as f64 / 1e3_f64;
    }
    return nan;
}

fn rollup_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    values.iter().fold(0.0,|r, x| r + *x)
}

fn rollup_rate_over_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change since rfa.prev_value.
        return 0.0;
    }
    let mut dt = rfa.window;
    if !rfa.prev_value.is_nan() {
        dt = timestamps[timestamps.len() - 1] - rfa.prev_timestamp
    }
    let sum = rfa.values.iter().fold(0.0,|r, x| r + *x);
    return sum / (dt as f64 / 1e3_f64);
}

fn rollup_range(rfa: &mut RollupFuncArg) -> f64 {
    let max = rollup_max(rfa);
    let min = rollup_min(rfa);
    return max - min;
}

fn rollup_sum2(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        return rfa.prev_value * rfa.prev_value;
    }
    values.iter().fold(0.0,|r, x| r + (*x * *x))
}

fn rollup_geomean(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let len = values.len();
    if len == 0 {
        return rfa.prev_value;
    }

    let p = values.iter().fold(1.0,|r, v| r * *v);
    return p.powf((1 / len) as f64);
}

fn rollup_absent(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.values.len() == 0 {
        return 1.0;
    }
    return nan;
}

fn rollup_present(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() > 0 {
        return 1.0;
    }
    return nan;
}

fn rollup_count(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        return nan;
    }
    return values.len() as f64;
}

fn rollup_stale_samples(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        return nan;
    }
    let mut n = 0;
    for v in rfa.values.iter() {
        if is_stale_nan(*v) {
            n += 1;
        }
    }
    return n as f64;
}

fn rollup_stddev(rfa: &mut RollupFuncArg) -> f64 {
    let std_var = rollup_stdvar(rfa);
    return std_var.sqrt();
}

fn rollup_stdvar(rfa: &mut RollupFuncArg) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
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
        count -= 1;
        let avg_new = avg + (*v - avg) / count as f64;
        q += (*v - avg) * (*v - avg_new);
        avg = avg_new
    }
    return q / count as f64;
}

fn rollup_increase_pure(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    // restore to the real value because of potential staleness reset
    let mut prev_value = rfa.real_prev_value;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        // Assume the counter starts from 0.
        prev_value = 0.0;
    }
    if values.len() == 0 {
        // Assume the counter didn't change since prev_value.
        return 0 as f64;
    }
    return values[values.len() - 1] - prev_value;
}

fn rollup_delta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    let mut prev_value = rfa.prev_value;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        if !rfa.real_prev_value.is_nan() {
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
        } else if !rfa.real_next_value.is_nan() {
            d = rfa.real_next_value - values[0]
        }
        if d == 0.0 {
            d = 10.0;
        }
        if values[0].abs() < 10.0 * (d.abs() + 1.0) {
            prev_value = 0.0;
        } else {
            prev_value = values[0];
            values = &values[1..]
        }
    }
    if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    return values[values.len() - 1] - prev_value;
}

fn rollup_delta_prometheus(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    // Just return the difference between the last and the first sample like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if values.len() < 2 {
        return nan;
    }
    return values[values.len() - 1] - values[0];
}

fn rollup_idelta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let last_value = values[values.len() - 1];
    let values = &values[0..values.len() - 1];
    if values.len() == 0 {
        let mut prev_value = rfa.prev_value;
        if prev_value.is_nan() {
            // Assume that the previous non-existing value was 0.
            return last_value;
        }
        return last_value - prev_value;
    }
    return last_value - values[values.len() - 1];
}

fn rollup_deriv_slow(rfa: &mut RollupFuncArg) -> f64 {
    // Use linear regression like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/73
    let (_, k) = linear_regression(rfa);
    return k;
}

fn rollup_deriv_fast(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let timestamps = &rfa.timestamps;
    let mut prev_value = rfa.prev_value;
    let mut prev_timestamp = rfa.prev_timestamp;
    if prev_value.is_nan() {
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
        prev_value = values[0];
        prev_timestamp = timestamps[0];
    } else if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];
    let dv = v_end - prev_value;
    let dt = (t_end - prev_timestamp) as f64 / 1e3_f64;
    return dv / dt;
}

fn rollup_ideriv(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let timestamps = &rfa.timestamps;
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
        return (values[0] as f64 - rfa.prev_value as f64) / ((timestamps[0] - rfa.prev_timestamp) as f64 / 1e3_f64);
    }
    let count = values.len();
    let v_end = values[count - 1];
    let t_end = timestamps[count - 1];
    let values = &values[0..count - 1];
    let mut timestamps = &timestamps[0..count - 1];
    // Skip data points with duplicate timestamps.
    while timestamps.len() > 0 && timestamps[count - 1] >= t_end {
        timestamps = &timestamps[0 .. count - 1];
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
        t_start = timestamps[count - 1];
        v_start = values[count - 1];
    }
    let dv = v_end - v_start;
    let dt = t_end - t_start;
    return dv / (dt as f64 / 1e3_f64);
}

fn rollup_lifetime(rfa: &mut RollupFuncArg) -> f64 {
    // Calculate the duration between the first and the last data points.
    let timestamps = &rfa.timestamps;
    if rfa.prev_value.is_nan() {
        if timestamps.len() < 2 {
            return nan;
        }
        return (timestamps[timestamps.len() - 1] as f64 - timestamps[0] as f64) / 1e3_f64;
    }
    if timestamps.len() == 0 {
        return nan;
    }
    return (timestamps[timestamps.len() - 1] as f64 - rfa.prev_timestamp as f64) / 1e3_f64;
}

fn rollup_lag(rfa: &mut RollupFuncArg) -> f64 {
    // Calculate the duration between the current timestamp and the last data point.
    let timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return (rfa.curr_timestamp - rfa.prev_timestamp) as f64 / 1e3_f64;
    }
    return (rfa.curr_timestamp - timestamps[timestamps.len() - 1]) as f64 / 1e3_f64;
}

fn rollup_scrape_interval(rfa: &mut RollupFuncArg) -> f64 {
    // Calculate the average interval between data points.
    let timestamps = &rfa.timestamps;
    if rfa.prev_value.is_nan() {
        if timestamps.len() < 2 {
            return nan;
        }
        return ((timestamps[timestamps.len() - 1] - timestamps[0]) as f64 / 1e3_f64) / (timestamps.len() - 1) as f64;
    }
    if timestamps.len() == 0 {
        return nan;
    }
    return ((timestamps[timestamps.len() - 1] - rfa.prev_timestamp) as f64 / 1e3_f64) / timestamps.len() as f64;
}

fn rollup_changes_prometheus(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    // do not take into account rfa.prev_value like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if values.len() < 1 {
        return nan;
    }
    let mut prev_value = values[0];
    let mut n = 0;
    for v in &values[1..] {
        if *v != prev_value {
            n += 1;
            prev_value = *v
        }
    }
    return n as f64;
}

fn rollup_changes(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut prev_value = &rfa.prev_value;
    let mut n = 0;
    let mut start = 0;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = &values[0];
        start = 1;
        n += 1;
    }
    for i in start .. values.len(){
        let v = values[i];
        if v != *prev_value {
            n += 1;
            prev_value = &v;
        }
    }
    return n as f64;
}

fn rollup_increases(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    let mut prev_value = rfa.prev_value;
    let mut cursor = values.as_slice();
    if prev_value.is_nan() {
        prev_value = cursor[0];
        cursor = &cursor[1..];
    }
    if cursor.len() == 0 {
        return 0.0;
    }
    let mut n = 0;
    for v in cursor.iter() {
        if v > &prev_value {
            n += 1;
        }
        prev_value = *v;
    }
    return n as f64;
}

// `decreases_over_time` logic is the same as `resets` logic.
const rollup_decreases: RollupFunc = rollup_resets;

fn rollup_resets(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    let mut prev_value = rfa.prev_value;
    let mut start: usize = 0;
    if prev_value.is_nan() {
        prev_value = values[0];
        start = 1;
    }
    if values.len() - start == 0 {
        return 0.0;
    }
    let mut n = 0;
    for cursor in start .. values.len() {
        let v = values[cursor];
        if v < prev_value {
            n += 1;
        }
        prev_value = v;
    }
    return n as f64;
}


/// get_candlestick_values returns a subset of rfa.values suitable for rollup_candlestick
///
/// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309 for details.
fn get_candlestick_values(rfa: &mut RollupFuncArg) -> &[f64] {
    let curr_timestamp = &rfa.curr_timestamp;
    let mut timestamps = &rfa.timestamps;
    let mut i = timestamps.len() - 1;

    while i >= 0 && timestamps[i] >= *curr_timestamp {
        i -= 1
    }

    if i == -1 {
        return &[];
    }

    return &rfa.values[0..i];
}

fn get_first_value_for_candlestick(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    return nan;
}

fn rollup_open(rfa: &mut RollupFuncArg) -> f64 {
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

fn rollup_close(rfa: &mut RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.len() == 0 {
        return get_first_value_for_candlestick(rfa);
    }
    values[values.len()]
}

fn rollup_high(rfa: &mut RollupFuncArg) -> f64 {
    let mut values = get_candlestick_values(rfa);
    let mut max = get_first_value_for_candlestick(rfa);
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
        if *v > max {
            max = *v
        }
    }
    return max;
}

fn rollup_low(rfa: &mut RollupFuncArg) -> f64 {
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
        if v < &min {
            min = *v
        }
    }
    return min;
}


fn rollup_mode_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // Copy rfa.values to a, since modeNoNaNs modifies a contents.
    let mut a = get_float64s(rfa.values.len());
    a.extend(&rfa.values);
    mode_no_nans(rfa.prev_value, &mut a)
}

fn rollup_ascent_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut prev_value = rfa.prev_value;
    let mut start: usize = 0;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = values[0];
        start = 1;
    }
    let mut s: f64 = 0.0;
    for i in start ..= values.len() - 1 {
        let v = values[i];
        let d = v - prev_value;
        if d > 0.0 {
            s += d;
        }
        prev_value = v;
    }
    return s;
}

fn rollup_descent_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    let mut prev_value = &rfa.prev_value;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = &values[0];
        values = &values[1..]
    }
    let mut s: f64 = 0.0;
    for v in values {
        let d = prev_value - v;
        if d > 0.0 {
            s += d
        }
        prev_value = v
    }
    return s;
}

fn rollup_zscore_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // See https://about.gitlab.com/blog/2019/07/23/anomaly-detection-using-prometheus/#using-z-score-for-anomaly-detection
    let scrape_interval = rollup_scrape_interval(rfa);
    let lag = rollup_lag(rfa);
    if scrape_interval.is_nan() || lag.is_nan() || lag > scrape_interval {
        return nan;
    }
    let d = rollup_last(rfa) - rollup_avg(rfa);
    if d == 0.0 {
        return 0.0;
    }
    return d / rollup_stddev(rfa);
}

fn rollup_first(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return values[0];
}

pub(crate) fn rollup_default(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    // Intentionally do not skip the possible last Prometheus staleness mark.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1526 .
    return *values.last().unwrap();
}

fn rollup_last(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return nan;
    }
    return *values.last().unwrap() as f64;
}

fn rollup_distinct(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return nan;
        }
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(&b));
    values.dedup();

    return values.len() as f64;
}


fn rollup_integrate(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns. 
    let mut values = &rfa.values[0..];
    let mut timestamps = &rfa.timestamps[0..];
    let mut prev_value = &rfa.prev_value;
    let mut prev_timestamp = &rfa.curr_timestamp - &rfa.window;
    if prev_value.is_nan() {
        if values.len() == 0 {
            return nan;
        }
        prev_value = &values[0];
        prev_timestamp = timestamps[0];
        values = &values[1..];
        timestamps = &timestamps[1..];
    }
    let mut sum: f64 = 0.0;
    for (i, v) in values.iter().enumerate() {
        let timestamp = timestamps[i];
        let dt = (timestamp - prev_timestamp) as f64 / 1e3_f64;
        sum = sum + prev_value * dt;
        prev_timestamp = timestamp;
        prev_value = v;
    }
    let dt = (&rfa.curr_timestamp - prev_timestamp) as f64 / 1e3_f64;
    sum = prev_value * dt;
    return sum;
}

fn rollup_fake(_rfa: &mut RollupFuncArg) -> f64 {
    panic!("BUG: rollup_fake shouldn't be called");
}

fn get_scalar(args: &Vec<RollupArgValue>, arg_num: usize) -> RuntimeResult<&Vec<f64>> {
    match args.get(arg_num) {
        Some(ts) => {
            if ts.len() != 1 {
                let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, ts.len());
                Err(RuntimeError::ArgumentError(msg))
            }
            Ok(&ts[0].values)
        }
        None => {
            let msg = format!("Argument index #{} of range;", arg_num + 1);
            Err(RuntimeError::ArgumentError(msg))
        }
    }
}

fn get_int_number(args: &Vec<RollupArgValue>, arg_num: usize) -> RuntimeResult<i64> {
    let v = get_scalar(args, arg_num)?;
    let mut n = 0;
    if v.len() > 0 {
        n = v[0] as i64;
    }
    return Ok(n);
}

pub(crate) fn get_string(tss: &[Timeseries], arg_num: usize) -> RuntimeResult<String> {
    if tss.len() != 1 {
        let msg = format!("arg # {} must contain a single timeseries; got {} timeseries", arg_num + 1, tss.len());
        return Err(RuntimeError::ArgumentError(msg));
    }
    let ts = &tss[0];
    for v in ts.values {
        if !v.is_nan() {
            let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
    // todo: COW
    return Ok(ts.metric_name.metric_group);
}

#[inline]
fn expect_rollup_args_num(args: &[RollupArgValue], expected_num: usize) -> RuntimeResult<()> {
    if args.len() == expected_num {
        return Ok(());
    }
    let msg = format!("unexpected number of args; got {}; want {}", args.len(), expected_num);
    Err(RuntimeError::ArgumentError(msg))
}

fn expect_at_least_n_args(tfa: &[RollupArgValue], n: usize) -> RuntimeResult<()> {
    let len = tfa.len();
    if len < n {
        let err = format!("not enough args; got {}; want at least {}", len, n);
        return Err(RuntimeError::ArgumentError(err));
    }
    Ok(())
}

static ROLLUP_FUNC_ARG_POOL: Lazy<LinearObjectPool<RollupFuncArg>> = Lazy::new(|| {
    LinearObjectPool::<RollupFuncArg>::new(
        || RollupFuncArg::default(),
        |v| {
            v.reset()
        }
    )
});


fn get_rollup_func_arg<'a>() -> LinearReusable<'a, RollupFuncArg> {
  ROLLUP_FUNC_ARG_POOL.pull()
}