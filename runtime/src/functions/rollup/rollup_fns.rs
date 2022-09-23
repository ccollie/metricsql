use std::default::Default;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use std::sync::Arc;

use lib::{get_float64s, is_stale_nan};
use metricsql::ast::{Expression, ExpressionNode};
use metricsql::functions::{can_adjust_window, RollupFunction};

use crate::{get_timeseries, get_timestamps};
use crate::eval::validate_max_points_per_timeseries;
use crate::functions::{mode_no_nans, quantile, quantiles};
use crate::functions::rollup::{RollupFn, RollupFunc, RollupFuncArg, RollupHandler, RollupHandlerEnum};
use crate::functions::types::ParameterValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};

use super::timeseries_map::TimeseriesMap;

// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/app/vmselect/promql/rollup.go

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

/// Maximum number of arguments permitted in a rollup function. This really only applies
/// to variadic functions like `aggr_over_time` and `quantiles_over_time`
const MAX_ARG_COUNT: usize = 32;

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


// Pre-allocated handlers for closure to save allocations at runtime
macro_rules! wrap_rollup_fn {
    ( $name: ident, $rf: expr ) => {
        const $name: RollupHandlerEnum = RollupHandlerEnum::wrap($rf);
    };
}

macro_rules! make_factory {
    ( $name: ident, $rf: expr ) => {
        #[inline]
        fn $name(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
            RollupHandlerEnum::wrap($rf)
        }
    };
}

macro_rules! fake_wrapper {
    ( $funcName: ident, $name: expr ) => {
        #[inline]
        fn $funcName(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
            RollupHandlerEnum::fake($name)
        }
    };
}


fn get_aggr_fn(f: RollupFunction) -> RuntimeResult<RollupFunc> {
    use RollupFunction::*;

    let res = match f {
        AbsentOverTime => rollup_absent,
        AscentOverTime => rollup_ascent_over_time,
        AvgOverTime => rollup_avg,
        Changes => rollup_changes,
        CountOverTime => rollup_count,
        DecreasesOverTime => rollup_decreases,
        DefaultRollup => rollup_default,
        Delta => rollup_delta,
        Deriv => rollup_deriv_slow,
        DerivFast => rollup_deriv_fast,
        DescentOverTime => rollup_descent_over_time,
        DistinctOverTime => rollup_distinct,
        FirstOverTime => rollup_first,
        GeomeanOverTime => rollup_geomean,
        IDelta => rollup_idelta,
        IDeriv => rollup_ideriv,
        Increase => rollup_delta,
        IncreasePure => rollup_increase_pure,
        IncreasesOverTime => rollup_increases,
        Integrate => rollup_integrate,
        IRate => rollup_ideriv,
        Lag => rollup_lag,
        LastOverTime => rollup_last,
        Lifetime => rollup_lifetime,
        MaxOverTime => rollup_max,
        MinOverTime => rollup_min,
        MedianOverTime => rollup_median,
        ModeOverTime => rollup_mode_over_time,
        PresentOverTime => rollup_present,
        RangeOverTime => rollup_range,
        Rate => rollup_deriv_fast,
        RateOverSum => rollup_rate_over_sum,
        Resets => rollup_resets,
        ScrapeInterval => rollup_scrape_interval,
        StaleSamplesOverTime => rollup_stale_samples,
        StddevOverTime => rollup_stddev,
        StdvarOverTime => rollup_stdvar,
        SumOverTime => rollup_sum,
        Sum2OverTime => rollup_sum2,
        TFirstOverTime => rollup_tfirst,
        Timestamp => rollup_tlast,
        TimestampWithName => rollup_tlast,
        TLastChangeOverTime => rollup_tlast_change,
        TLastOverTime => rollup_tlast,
        TMaxOverTime => rollup_tmax,
        TMinOverTime => rollup_tmin,
        ZScoreOverTime => rollup_zscore_over_time,
        _ => return Err(RuntimeError::General(
            format!("{} is not an aggregate function", f.name())
        ))
    };

    Ok(res)
}

fn get_aggr_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
    match RollupFunction::from_str(name) {
        Err(e) => Err(RuntimeError::UnknownFunction(name.to_string())),
        Ok(func) => Ok(func)
    }
}

fn requires_config(f: RollupFunction) -> bool {
    use RollupFunction::*;

    matches!(f,
        PredictLinear | DurationOverTime | HoltWinters |
        ShareLeOverTime | ShareGtOverTime |
        CountLeOverTime | CountGtOverTime | CountEqOverTime | CountNeOverTime |
        HoeffdingBoundLower | HoeffdingBoundUpper |
        QuantilesOverTime | QuantileOverTime
    )
}

fn is_aggr_function(f: RollupFunction) -> bool {
    use RollupFunction::*;
    matches!(f,
        AbsentOverTime | AscentOverTime | AvgOverTime |
        Changes | CountOverTime | DecreasesOverTime |
        DefaultRollup | Delta | Deriv | DerivFast |
        DescentOverTime | DistinctOverTime | FirstOverTime |
        GeomeanOverTime | IDelta | IDeriv | Increase |
        IncreasePure | IncreasesOverTime | Integrate | IRate |
        Lag | LastOverTime | Lifetime | MaxOverTime | MinOverTime |
        MedianOverTime | ModeOverTime | PresentOverTime | RangeOverTime |
	    Rate | RateOverSum | Resets | ScrapeInterval | StaleSamplesOverTime |
	    StddevOverTime | StdvarOverTime | SumOverTime | Sum2OverTime |
	    TFirstOverTime | Timestamp | TimestampWithName | TLastChangeOverTime |
	    TLastOverTime | TMaxOverTime | TMinOverTime | ZScoreOverTime
    )
}


// pass through to existing functions

///////////
make_factory!(new_rollup_absent_over_time,      rollup_absent);
make_factory!(new_rollup_aggr_over_time,        rollup_fake);
make_factory!(new_rollup_ascent_over_time,      rollup_ascent_over_time);
make_factory!(new_rollup_avg_over_time,         rollup_avg);
make_factory!(new_rollup_changes,               rollup_changes);
make_factory!(new_rollup_changes_prometheus,    rollup_changes_prometheus);
make_factory!(new_rollup_count_over_time,       rollup_count);
make_factory!(new_rollup_decreases_over_time,   rollup_decreases);
make_factory!(new_rollup_default,               rollup_default);
make_factory!(new_rollup_delta,                 rollup_delta);
make_factory!(new_rollup_delta_prometheus,      rollup_delta_prometheus);
make_factory!(new_rollup_deriv,                 rollup_deriv_slow);
make_factory!(new_rollup_deriv_fast,            rollup_deriv_fast);
make_factory!(new_rollup_descent_over_time,     rollup_descent_over_time);
make_factory!(new_rollup_distinct_over_time,    rollup_distinct);
make_factory!(new_rollup_first_over_time,       rollup_first);
make_factory!(new_rollup_geomean_over_time,     rollup_geomean);
make_factory!(new_rollup_histogram_over_time,   rollup_histogram);
make_factory!(new_rollup_idelta,                rollup_idelta);
make_factory!(new_rollup_ideriv,                rollup_ideriv);
make_factory!(new_rollup_increase,              rollup_delta);
make_factory!(new_rollup_increase_pure,         rollup_increase_pure);
make_factory!(new_rollup_increases_over_time,   rollup_increases);
make_factory!(new_rollup_integrate,             rollup_integrate);
make_factory!(new_rollup_irate,                 rollup_ideriv);
make_factory!(new_rollup_lag,                   rollup_lag);
make_factory!(new_rollup_last_over_time,        rollup_last);
make_factory!(new_rollup_lifetime,              rollup_lifetime);
make_factory!(new_rollup_max_over_time,         rollup_max);
make_factory!(new_rollup_min_over_time,         rollup_min);
make_factory!(new_rollup_median_over_time,      rollup_median);
make_factory!(new_rollup_mode_over_time,        rollup_mode_over_time);
make_factory!(new_rollup_present_over_time,     rollup_present);
make_factory!(new_rollup_range_over_time,       rollup_range);
make_factory!(new_rollup_rate,                  rollup_deriv_fast);
make_factory!(new_rollup_rate_over_sum,         rollup_rate_over_sum);
make_factory!(new_rollup_resets,                rollup_resets);
make_factory!(new_rollup_scrape_interval,       rollup_scrape_interval);
make_factory!(new_rollup_stale_samples_over_time, rollup_stale_samples);
make_factory!(new_rollup_stddev_over_time,      rollup_stddev);
make_factory!(new_rollup_stdvar_over_time,      rollup_stdvar);
make_factory!(new_rollup_sum_over_time,         rollup_sum);
make_factory!(new_rollup_sum2_over_time,        rollup_sum2);
make_factory!(new_rollup_tfirst_over_time,      rollup_tfirst);
make_factory!(new_rollup_timestamp,             rollup_tlast);
make_factory!(new_rollup_timestamp_with_name,   rollup_tlast);
make_factory!(new_rollup_tlast_change_over_time, rollup_tlast_change);
make_factory!(new_rollup_tlast_over_time,       rollup_tlast);
make_factory!(new_rollup_tmax_over_time,        rollup_tmax);
make_factory!(new_rollup_tmin_over_time,        rollup_tmin);
make_factory!(new_rollup_zscore_over_time,      rollup_zscore_over_time);

fake_wrapper!(new_rollup,                   "rollup");
fake_wrapper!(new_rollup_candlestick,       "rollup_candlestick");
fake_wrapper!(new_rollup_rollup_delta,      "rollup_delta");
fake_wrapper!(new_rollup_rollup_deriv,      "rollup_deriv");
fake_wrapper!(new_rollup_rollup_increase,   "rollup_increase"); // + rollupFuncsRemoveCounterResets
fake_wrapper!(new_rollup_rollup_rate,       "rollup_rate"); // + rollupFuncsRemoveCounterResets
fake_wrapper!(new_rollup_rollup_scrape_interval, "rollup_scrape_interval");



pub(crate) fn get_rollup_function_factory(func: RollupFunction) -> fn(&Vec<ParameterValue>) -> RollupHandlerEnum {
    let imp = match func {
        RollupFunction::AbsentOverTime => new_rollup_absent_over_time,
        RollupFunction::AggrOverTime => new_rollup_aggr_over_time,
        RollupFunction::AscentOverTime => new_rollup_ascent_over_time,
        RollupFunction::AvgOverTime => new_rollup_avg_over_time,
        RollupFunction::Changes => new_rollup_changes,
        RollupFunction::ChangesPrometheus => new_rollup_changes_prometheus,
        RollupFunction::CountEqOverTime => new_rollup_count_eq,
        RollupFunction::CountGtOverTime => new_rollup_count_gt,
        RollupFunction::CountLeOverTime => new_rollup_count_le,
        RollupFunction::CountNeOverTime => new_rollup_count_ne,
        RollupFunction::CountOverTime => new_rollup_count_over_time,
        RollupFunction::DecreasesOverTime => new_rollup_decreases_over_time,
        RollupFunction::DefaultRollup => new_rollup_default,
        RollupFunction::Delta => new_rollup_delta,
        RollupFunction::DeltaPrometheus => new_rollup_delta_prometheus,
        RollupFunction::Deriv => new_rollup_deriv,
        RollupFunction::DerivFast => new_rollup_deriv_fast,
        RollupFunction::DescentOverTime => new_rollup_descent_over_time,
        RollupFunction::DistinctOverTime => new_rollup_distinct_over_time,
        RollupFunction::DurationOverTime => new_rollup_duration_over_time,
        RollupFunction::FirstOverTime => new_rollup_first_over_time,
        RollupFunction::GeomeanOverTime => new_rollup_geomean_over_time,
        RollupFunction::HistogramOverTime => new_rollup_histogram_over_time,
        RollupFunction::HoeffdingBoundLower => new_rollup_hoeffding_bound_lower,
        RollupFunction::HoeffdingBoundUpper => new_rollup_hoeffding_bound_upper,
        RollupFunction::HoltWinters => new_rollup_holt_winters,
        RollupFunction::IDelta => new_rollup_idelta,
        RollupFunction::IDeriv => new_rollup_ideriv,
        RollupFunction::Increase => new_rollup_increase,
        RollupFunction::IncreasePrometheus => new_rollup_delta_prometheus,
        RollupFunction::IncreasePure => new_rollup_increase_pure,
        RollupFunction::IncreasesOverTime => new_rollup_increases_over_time,
        RollupFunction::Integrate => new_rollup_integrate,
        RollupFunction::IRate => new_rollup_irate,
        RollupFunction::Lag => new_rollup_lag,
        RollupFunction::LastOverTime => new_rollup_last_over_time,
        RollupFunction::Lifetime => new_rollup_lifetime,
        RollupFunction::MaxOverTime => new_rollup_max_over_time,
        RollupFunction::MedianOverTime => new_rollup_median_over_time,
        RollupFunction::MinOverTime => new_rollup_min_over_time,
        RollupFunction::ModeOverTime => new_rollup_mode_over_time,
        RollupFunction::PredictLinear => new_rollup_predict_linear,
        RollupFunction::PresentOverTime => new_rollup_present_over_time,
        RollupFunction::QuantileOverTime => new_rollup_quantile,
        RollupFunction::QuantilesOverTime => new_rollup_quantiles,
        RollupFunction::RangeOverTime => new_rollup_range_over_time,
        RollupFunction::Rate => new_rollup_rate,
        RollupFunction::RateOverSum => new_rollup_rate_over_sum,
        RollupFunction::Resets => new_rollup_resets,
        RollupFunction::Rollup => new_rollup,
        RollupFunction::RollupCandlestick => new_rollup_candlestick,
        RollupFunction::RollupDelta => new_rollup_delta,
        RollupFunction::RollupDeriv => new_rollup_deriv,
        RollupFunction::RollupIncrease => new_rollup_increase,
        RollupFunction::RollupRate => new_rollup_rate,
        RollupFunction::RollupScrapeInterval => new_rollup_scrape_interval,
        RollupFunction::ScrapeInterval => new_rollup_scrape_interval,
        RollupFunction::ShareGtOverTime => new_rollup_share_gt,
        RollupFunction::ShareLeOverTime => new_rollup_share_le,
        RollupFunction::StaleSamplesOverTime => new_rollup_stale_samples_over_time,
        RollupFunction::StddevOverTime => new_rollup_stddev_over_time,
        RollupFunction::StdvarOverTime => new_rollup_stdvar_over_time,
        RollupFunction::SumOverTime => new_rollup_sum_over_time,
        RollupFunction::Sum2OverTime => new_rollup_sum2_over_time,
        RollupFunction::TFirstOverTime => new_rollup_first_over_time,
        RollupFunction::Timestamp => new_rollup_timestamp,
        RollupFunction::TimestampWithName => new_rollup_timestamp_with_name,
        RollupFunction::TLastChangeOverTime => new_rollup_tlast_change_over_time,
        RollupFunction::TLastOverTime => new_rollup_tlast_over_time,
        RollupFunction::TMaxOverTime => new_rollup_tmax_over_time,
        RollupFunction::TMinOverTime => new_rollup_tmin_over_time,
        RollupFunction::ZScoreOverTime => new_rollup_zscore_over_time
    };

    return imp;
}


#[inline]
fn is_removes_counter_resets(f: RollupFunction) -> bool {
    use RollupFunction::*;
    matches!(f, Increase | IncreasePrometheus | IncreasePure | IRate | RollupIncrease | RollupRate )
}

pub(crate) fn rollup_func_keeps_metric_name(name: &str) -> bool {
    match RollupFunction::from_str(name) {
        Err(_) => false,
        Ok(func) => func.keep_metric_name()
    }
}


// todo: use in optimize so its cached in the ast
pub(crate) fn get_rollup_aggr_funcs(expr: &Expression) -> RuntimeResult<Vec<RollupFunction>> {
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
            if fe_.name() != "aggr_over_time" {
                let msg = format!("BUG: unexpected function name: {}; want `aggr_over_time`", fe_.name());
                return Err(RuntimeError::from(msg));
            }

            let arg_len = fe_.args.len();
            if arg_len < 2 {
                let msg = format!("unexpected number of args to aggr_over_time(); got {}; want at least {}", arg_len, 2);
                return Err(RuntimeError::from(msg));
            }

            let mut aggr_funcs: Vec<RollupFunction> = Vec::with_capacity(1);
            for i in 0..fe_.args.len() - 1 {
                let arg = &fe_.args[i];
                match arg.deref() {
                    Expression::String(se) => {
                        let name = &se.s;
                        match get_aggr_func_by_name(name) {
                            Err(_) => {
                                let msg = format!("{} cannot be used in `aggr_over_time` function; expecting quoted aggregate function name", name);
                                return Err(RuntimeError::General(msg));
                            }
                            Ok(rf) => {
                                aggr_funcs.push(rf)
                            }
                        }
                    },
                    _ => {
                        let msg = format!("{} cannot be passed here; expecting quoted aggregate function name", arg);
                        return Err(RuntimeError::General(msg));
                    }
                }
            }

            Ok(aggr_funcs)
        }
    }

}


pub(crate) type PreFunction = fn(&mut [f64], &[i64]) -> ();

#[inline]
pub(crate) fn eval_prefuncs(fns: &Vec<PreFunction>, values: &mut [f64], timestamps: &[i64]) {
    for f in fns {
        f(values, timestamps)
    }
}

#[inline]
fn remove_counter_resets_pre_func(values: &mut [f64], _: &[i64]) {
    remove_counter_resets(values);
}

#[inline]
fn delta_values_pre_func(values: &mut [f64], _: &[i64]) -> () {
    delta_values(values);
}

/// Calculate intervals in seconds between samples.
fn calc_sample_intervals_pre_fn(values: &mut [f64], timestamps: &[i64]) {
    // Calculate intervals in seconds between samples.
    let mut ts_secs_prev = NAN;
    for (i, ts) in timestamps.iter().enumerate() {
        let ts_secs = (ts / 1000) as f64;
        values[i] = ts_secs - ts_secs_prev;
        ts_secs_prev = ts_secs;
    };
    if values.len() > 1 {
        // Overwrite the first NaN interval with the second interval,
        // So min, max and avg rollup could be calculated properly,
        // since they don't expect to receive NaNs.
        values[0] = values[1]
    }
}


wrap_rollup_fn!(fn_open,   rollup_open);
wrap_rollup_fn!(fn_close,  rollup_close);
wrap_rollup_fn!(fn_min,    rollup_min);
wrap_rollup_fn!(fn_max,    rollup_max);
wrap_rollup_fn!(fn_avg,    rollup_avg);
wrap_rollup_fn!(fn_low,    rollup_low);
wrap_rollup_fn!(fn_high,   rollup_high);
wrap_rollup_fn!(fn_fake,   rollup_fake);

pub(crate) fn get_rollup_configs<'a>(
    func: &RollupFunction,
    rf: &'a RollupHandlerEnum,
    expr: &Expression,
    start: i64, end: i64, step: i64, window: i64,
    max_points_per_series: usize,
    lookback_delta: i64,
    shared_timestamps: &Arc<Vec<i64>>) -> RuntimeResult<(Vec<RollupConfig<'a>>, Vec<PreFunction>)> {

    // todo: use tinyvec
    let mut pre_funcs: Vec<PreFunction> = Vec::with_capacity(3);

    if func.should_remove_counter_resets() {
        pre_funcs.push(remove_counter_resets_pre_func);
    }

    let may_adjust_window = can_adjust_window(func);
    let is_default_rollup = *func == RollupFunction::DefaultRollup;

    let template = RollupConfig {
        tag_value: "".to_string(),
        handler: &fn_fake,
        start,
        end,
        step,
        window,
        may_adjust_window,
        lookback_delta,
        timestamps: shared_timestamps.clone(),
        is_default_rollup,
        max_points_per_timeseries: max_points_per_series,
    };

    let new_rollup_config = |rf: &RollupHandlerEnum, tag_value: &str| -> RollupConfig {
        template.clone_with_fn(rf, tag_value)
    };
    
    let append_rollup_configs = |dst: &mut Vec<RollupConfig>| {
        dst.push(new_rollup_config(&fn_min, "min"));
        dst.push(new_rollup_config(&fn_max, "max"));
        dst.push(new_rollup_config(&fn_avg, "avg"));
    };

    // todo: tinyvec
    let mut rcs: Vec<RollupConfig> = Vec::with_capacity(1);
    match func {
        RollupFunction::Rollup => append_rollup_configs(&mut rcs),
        RollupFunction::RollupRate | RollupFunction::RollupDeriv => {
            pre_funcs.push(delta_values_pre_func);
            append_rollup_configs(&mut rcs);
        },
        RollupFunction::RollupIncrease | RollupFunction::RollupDelta => {
            pre_funcs.push(delta_values_pre_func);
            append_rollup_configs(&mut rcs);
        },
        RollupFunction::RollupCandlestick => {
            rcs.push(new_rollup_config(&fn_open, "open"));
            rcs.push(new_rollup_config(&fn_close, "close"));
            rcs.push(new_rollup_config(&fn_low, "low"));
            rcs.push(new_rollup_config(&fn_high, "high"));
        },
        RollupFunction::RollupScrapeInterval => {
            pre_funcs.push(calc_sample_intervals_pre_fn);
            append_rollup_configs(&mut rcs);
        },
        RollupFunction::AggrOverTime => {
            match get_rollup_aggr_funcs(expr) {
                Err(_) => {
                    return Err(RuntimeError::ArgumentError(format!("invalid args to {}", expr)))
                },
                Ok(funcs) => {
                    for rf in funcs {
                        if is_removes_counter_resets(rf) {
                            // There is no need to save the previous pre_func, since it is either empty or the same.
                            pre_funcs.clear();
                            pre_funcs.push(remove_counter_resets_pre_func);
                        }
                        let rollup_fn = get_aggr_fn(rf)?;
                        let handler = RollupHandlerEnum::wrap(rollup_fn);
                        rcs.push(new_rollup_config(&handler, &rf.name()));
                    }
                }
            }
        },
        _ => {
            rcs.push(new_rollup_config(rf, ""));
        }
    }

    Ok((rcs, pre_funcs))
}


pub(crate) struct RollupConfig<'a> {
    /// This tag value must be added to "rollup" tag if non-empty.
    pub tag_value: String,
    handler: &'a RollupHandlerEnum,
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

impl<'a> RollupConfig<'a> {
    fn clone_with_fn(&self, rollup_fn: &RollupHandlerEnum, tag_value: &str) -> Self {
        return RollupConfig {
            tag_value: tag_value.to_string(),
            handler: rollup_fn,
            start: self.start,
            end: self.end,
            step: self.step,
            window: self.window,
            may_adjust_window: self.may_adjust_window,
            lookback_delta: self.lookback_delta,
            timestamps: self.timestamps.clone(),
            is_default_rollup: self.is_default_rollup,
            max_points_per_timeseries: self.max_points_per_timeseries,
        }
    }

    /// calculates rollup for the given timestamps and values, appends
    /// them to dst_values and returns results.
    ///
    /// rc.timestamps are used as timestamps for dst_values.
    ///
    /// timestamps must cover time range [rc.start - rc.window - MAX_SILENCE_INTERVAL ... rc.end].
    ///
    /// do cannot be called from concurrent goroutines.
    pub(crate) fn exec(&mut self, dst_values: &mut Vec<f64>, values: &[f64], timestamps: &[i64]) -> RuntimeResult<()> {
        self.do_internal(dst_values, None, values, timestamps)
    }

    /// calculates rollup for the given timestamps and values and puts them to tsm.
    pub(crate) fn do_timeseries_map(&mut self, tsm: &mut TimeseriesMap, values: &[f64], timestamps: &[i64]) -> RuntimeResult<()> {
        let mut ts = get_timeseries();
        self.do_internal(&mut ts.values,Some(tsm), values, timestamps)
    }

    pub(in crate::functions) fn get_timestamps(&self) -> RuntimeResult<Vec<i64>> {
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

        let mut rfa = RollupFuncArg::default();
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

            rfa.prev_value = NAN;
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
                rfa.real_prev_value = NAN;
            }
            if j < values.len() {
                rfa.real_next_value = values[j];
            } else {
                rfa.real_next_value = NAN;
            }
            rfa.curr_timestamp = *tEnd;
            let value = (self.handler).eval(&mut rfa);
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

fn new_rollup_holt_winters(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    // unwrap is sound since arguments are checked before this is called
    let sfs = args[1].get_vector().unwrap();
    let tfs = args[2].get_vector().unwrap();

    RollupHandlerEnum::General(Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        holt_winters_internal(rfa, sfs, tfs)
    }))
}

fn holt_winters_internal(rfa: &mut RollupFuncArg, sfs: &Vec<f64>, tfs: &Vec<f64>) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        return rfa.prev_value;
    }
    let sf = sfs[rfa.idx];
    if sf <= 0.0 || sf >= 1.0 {
        return NAN;
    }
    let tf = tfs[rfa.idx];
    if tf <= 0.0 || tf >= 1.0 {
        return NAN;
    }

    let mut ofs = 0;

    // See https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing .
    // TODO: determine whether this shit really works.
    let mut s0 = rfa.prev_value;
    if s0.is_nan() {
        ofs = 1;
        s0 = rfa.values[0];

        if rfa.values.len() <= 1 {
            return s0;
        }
    }

    let mut b0 = rfa.values[ofs] - s0;
    for i in ofs..rfa.values.len() {
        let v = rfa.values[i];
        let s1 = sf * v + (1.0 - sf) * (s0 + b0);
        let b1 = tf * (s1 - s0) + (1.0 - tf) * b0;
        s0 = s1;
        b0 = b1
    }

    s0
}

fn new_rollup_predict_linear(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let secs = args[1].get_vector().unwrap();

    RollupHandlerEnum::General(Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (v, k) = linear_regression(rfa);
        if v.is_nan() {
            return NAN;
        }
        return v + k * secs[rfa.idx];
    }))

}

fn linear_regression(rfa: &mut RollupFuncArg) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
    let n = values.len();
    if n == 0 {
        return (NAN, NAN);
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

    true
}

fn new_rollup_duration_over_time(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let d_maxs = args[1].get_vector().unwrap();

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.timestamps.len() == 0 {
            return NAN;
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

        (d_sum as f64 / 1000_f64) as f64
    });

    RollupHandlerEnum::General(f)
}

fn new_rollup_share_le(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_share_filter(args, count_filter_le)
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

fn new_rollup_share_gt(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_share_filter(args, count_filter_gt)
}

#[inline]
fn count_filter_gt(values: &[f64], gt: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if *v > gt {
            n += 1;
        }
    }
    n
}

#[inline]
fn count_filter_eq(values: &[f64], eq: f64) -> i32 {
    let mut n = 0;
    for v in values {
        if *v == eq {
            n += 1;
        }
    }
    n
}

#[inline]
fn count_filter_ne(values: &[f64], ne: f64) -> i32 {
    let mut n = 0;
    for v in values.iter() {
        if *v != ne {
            n += 1;
        }
    }
    n
}


fn new_rollup_share_filter(args: &Vec<ParameterValue>, count_filter: fn(values: &[f64], limit: f64) -> i32) -> RollupHandlerEnum {
    let rf = new_rollup_count_filter(args, count_filter);
    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let n = rf.eval(rfa);
        return n / rfa.values.len() as f64;
    });

    RollupHandlerEnum::General(f)
}

fn new_rollup_count_le(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_count_filter(args, count_filter_le)
}

fn new_rollup_count_gt(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_count_filter(args, count_filter_gt)
}

fn new_rollup_count_eq(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_count_filter(args, count_filter_eq)
}

fn new_rollup_count_ne(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    new_rollup_count_filter(args, count_filter_ne)
}

fn new_rollup_count_filter(
    args: &Vec<ParameterValue>,
    count_filter: fn(values: &[f64], limit: f64) -> i32) -> RollupHandlerEnum {

    // `unwrap()` is sound since parameters are checked before this function is called
    let limits = args[1].get_vector().unwrap();

    let handler = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.values.len() == 0 {
            return NAN;
        }
        let limit = limits[rfa.idx];
        count_filter(&rfa.values, limit as f64) as f64
    });

    RollupHandlerEnum::General(handler)
}

fn new_rollup_hoeffding_bound_lower(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let phis = args[0].get_vector().unwrap();

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(rfa, &phis);
        avg - bound
    });

    RollupHandlerEnum::General(f)
}

fn new_rollup_hoeffding_bound_upper(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let phis = args[0].get_vector().unwrap();

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(rfa, &phis);
        return avg + bound;
    });

    RollupHandlerEnum::General(f)
}

fn rollup_hoeffding_bound_internal(rfa: &mut RollupFuncArg, phis: &[f64]) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        return (NAN, NAN);
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
        return (INF, v_avg);
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

fn new_rollup_quantiles(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let phi_label = args[0].get_str().unwrap();
    let cap = args.len() - 1;

    let mut phis = Vec::with_capacity(cap);
    // todo: smallvec ??
    let mut phi_strs: Vec<String> = Vec::with_capacity(cap);

    for i in 1..args.len() {
        // unwrap should be safe, since parameters types are checked before calling the function
        let v = args[i].get_float().unwrap();
        phis[i] = v;
        phi_strs[i] = format!("{}", v);
    }

    let f: Box<dyn RollupFn> = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.values.len() == 0 {
            return rfa.prev_value;
        }
        if rfa.values.len() == 1 {
            // Fast path - only a single value.
            return rfa.values[0];
        }
        // tinyvec ?
        let mut qs = get_float64s(phis.len());
        quantiles(qs.deref_mut(), &phis, &rfa.values);
        let idx = rfa.idx;
        let mut tsm = rfa.tsm.unwrap();
        for (i, phiStr) in phi_strs.iter().enumerate() {
            let mut ts = tsm.get_or_create_timeseries(&phi_label, phiStr);
            ts.values[idx] = qs[i];
        }

        return NAN;
    });

    RollupHandlerEnum::General(f)
}

fn new_rollup_quantile(args: &Vec<ParameterValue>) -> RollupHandlerEnum {
    let phis = args[0].get_vector().unwrap();

    let rf = Box::new(move |rfa: &mut RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        let phi = phis[rfa.idx];
        quantile(phi, &rfa.values)
    });

    RollupHandlerEnum::General(rf)
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
    return NAN;
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
        return NAN;
    }
    let sum: f64 = values.iter().fold(0.0,|r, x| r + *x);
    return sum / values.len() as f64;
}

fn rollup_min(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    let mut min_value = rfa.values[0];
    for v in rfa.values {
        if v < min_value {
            min_value = v;
        }
    }

    min_value
}

fn rollup_max(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    *rfa.values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

fn rollup_median(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    quantile(0.5, &rfa.values)
}

fn rollup_tmin(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let mut timestamps = &rfa.timestamps;
    if values.len() == 0 {
        return NAN;
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

    if rfa.values.len() == 0 {
        return NAN;
    }

    let mut max_value = rfa.values[0];
    let mut max_timestamp = rfa.timestamps[0];
    for (i, v) in rfa.values.iter().enumerate() {
        // Get the last timestamp for the maximum value as most users expect.
        if *v >= max_value {
            max_value = *v;
            max_timestamp = rfa.timestamps[i];
        }
    }

    return (max_timestamp as f64 / 1e3_f64) as f64;
}

fn rollup_tfirst(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.timestamps.len() == 0 {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    return rfa.timestamps[0] as f64 / 1e3_f64;
}

fn rollup_tlast(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    return timestamps[timestamps.len() - 1] as f64 / 1e3_f64;
}

fn rollup_tlast_change(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    if values.len() == 0 {
        return NAN;
    }
    let mut timestamps = &rfa.timestamps;
    let last = values.len() - 1;
    let last_value = values[last];
    values = &values[0..last];
    let mut i = last;
    for value in rfa.values.iter().rev() {
        if *value != last_value {
            return timestamps[i + 1] as f64 / 1e3_f64;
        }
        i -= 1;
    }
    if rfa.prev_value.is_nan() || rfa.prev_value != last_value {
        return timestamps[0] as f64 / 1e3_f64;
    }
    return NAN;
}

fn rollup_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    if rfa.values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    rfa.values.iter().fold(0.0,|r, x| r + *x)
}

fn rollup_rate_over_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        // Assume that the value didn't change since rfa.prev_value.
        return 0.0;
    }
    let mut dt = rfa.window;
    if !rfa.prev_value.is_nan() {
        dt = timestamps[timestamps.len() - 1] - rfa.prev_timestamp
    }
    let sum = rfa.values.iter().fold(0.0,|r, x| r + *x);
    return sum / (dt / rfa.window) as f64;
}

fn rollup_range(rfa: &mut RollupFuncArg) -> f64 {
    let max = rollup_max(rfa);
    let min = rollup_min(rfa);
    return max - min;
}

fn rollup_sum2(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        return rfa.prev_value * rfa.prev_value;
    }
    rfa.values.iter().fold(0.0,|r, x| r + (*x * *x))
}

fn rollup_geomean(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let len = rfa.values.len();
    if len == 0 {
        return rfa.prev_value;
    }

    let p = rfa.values.iter().fold(1.0,|r, v| r * *v);
    return p.powf((1 / len) as f64);
}

fn rollup_absent(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.values.len() == 0 {
        return 1.0;
    }
    return NAN;
}

fn rollup_present(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() > 0 {
        return 1.0;
    }
    return NAN;
}

fn rollup_count(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        return NAN;
    }
    return rfa.values.len() as f64;
}

fn rollup_stale_samples(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        return NAN;
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
        return NAN;
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

    let count = rfa.values.len();
    // restore to the real value because of potential staleness reset
    if rfa.prev_value.is_nan() {
        if count == 0 {
            return NAN;
        }
        // Assume the counter starts from 0.
        rfa.prev_value = 0.0;
    }
    if rfa.values.len() == 0 {
        // Assume the counter didn't change since prev_value.
        return 0 as f64;
    }
    return rfa.values[count - 1] - rfa.prev_value;
}

fn rollup_delta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    if rfa.prev_value.is_nan() {
        if values.len() == 0 {
            return NAN;
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
        if rfa.values.len() > 1 {
            d = rfa.values[1] - rfa.values[0]
        } else if !rfa.real_next_value.is_nan() {
            d = rfa.real_next_value - values[0]
        }
        if d == 0.0 {
            d = 10.0;
        }
        if rfa.values[0].abs() < 10.0 * (d.abs() + 1.0) {
            rfa.prev_value = 0.0;
        } else {
            rfa.prev_value = rfa.values[0];
            values = &values[1..]
        }
    }
    if values.len() == 0 {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    return values[values.len() - 1] - rfa.prev_value;
}

fn rollup_delta_prometheus(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let count = rfa.values.len();
    // Just return the difference between the last and the first sample like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if count < 2 {
        return NAN;
    }
    return rfa.values[count - 1] - rfa.values[0];
}

fn rollup_idelta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let last_value = rfa.values[rfa.values.len() - 1];
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
            return NAN;
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
            // So just return NAN
            return NAN;
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
    let mut count = rfa.values.len();
    if count < 2 {
        if count == 0 {
            return NAN;
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
            // So just return NAN
            return NAN;
        }
        return (rfa.values[0] as f64 - rfa.prev_value as f64) / ((rfa.timestamps[0] - rfa.prev_timestamp) as f64 / 1e3_f64);
    }
    let v_end = rfa.values[count - 1];
    let t_end = rfa.timestamps[count - 1];
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
            return NAN;
        }
        return (timestamps[timestamps.len() - 1] as f64 - timestamps[0] as f64) / 1e3_f64;
    }
    if timestamps.len() == 0 {
        return NAN;
    }
    return (timestamps[timestamps.len() - 1] as f64 - rfa.prev_timestamp as f64) / 1e3_f64;
}

fn rollup_lag(rfa: &mut RollupFuncArg) -> f64 {
    // Calculate the duration between the current timestamp and the last data point.
    let count = rfa.timestamps.len();
    if count == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return (rfa.curr_timestamp - rfa.prev_timestamp) as f64 / 1e3_f64;
    }
    return (rfa.curr_timestamp - rfa.timestamps[count - 1]) as f64 / 1e3_f64;
}

fn rollup_scrape_interval(rfa: &mut RollupFuncArg) -> f64 {
    // Calculate the average interval between data points.
    let count = rfa.timestamps.len();
    if rfa.prev_value.is_nan() {
        if count < 2 {
            return NAN;
        }
        return ((rfa.timestamps[count - 1] - rfa.timestamps[0]) as f64 / 1e3_f64) / (count - 1) as f64;
    }
    if count == 0 {
        return NAN;
    }
    return ((rfa.timestamps[count - 1] - rfa.prev_timestamp) as f64 / 1e3_f64) / count as f64;
}

fn rollup_changes_prometheus(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // do not take into account rfa.prev_value like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if rfa.values.len() < 1 {
        return NAN;
    }
    let mut prev_value = rfa.values[0];
    let mut n = 0;
    for i in 1 .. rfa.values.len() {
        let v = rfa.values[i];
        if v != prev_value {
            n += 1;
            prev_value = v
        }
    }
    return n as f64;
}

fn rollup_changes(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut n = 0;
    let mut start = 0;
    if rfa.prev_value.is_nan() {
        if rfa.values.len() == 0 {
            return NAN;
        }
        rfa.prev_value = rfa.values[0];
        start = 1;
        n += 1;
    }
    for i in start .. rfa.values.len(){
        let v = rfa.values[i];
        if v != rfa.prev_value {
            n += 1;
            rfa.prev_value = v;
        }
    }
    return n as f64;
}

fn rollup_increases(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return 0.0;
    }

    let mut start = 0;
    if rfa.prev_value.is_nan() {
        rfa.prev_value = rfa.values[0];
        start = 1;
    }
    if rfa.values.len() == start {
        return 0.0;
    }
    let mut n = 0;
    for i in start .. rfa.values.len() {
        let v = rfa.values[i];
        if v > rfa.prev_value {
            n += 1;
        }
        rfa.prev_value = v;
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
            return NAN;
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
    let mut i = rfa.timestamps.len() - 1;

    loop {
        if i >= 0 && rfa.timestamps[i] >= *curr_timestamp {
            i -= 1
        } else {
            if i == 0 {
                return &[];
            }
            break
        }
    }

    return &rfa.values[0..i as usize];
}

fn get_first_value_for_candlestick(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    return NAN;
}

fn rollup_open(rfa: &mut RollupFuncArg) -> f64 {
    let v = get_first_value_for_candlestick(rfa);
    if !v.is_nan() {
        return v;
    }
    let values = get_candlestick_values(rfa);
    if values.len() == 0 {
        return NAN;
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
            return NAN;
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
            return NAN;
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
            return NAN;
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
    let mut ofs = 0;
    if rfa.prev_value.is_nan() {
        if rfa.values.len() == 0 {
            return NAN;
        }
        rfa.prev_value = rfa.values[0];
        ofs = 1;
    }

    let mut s: f64 = 0.0;
    for i in ofs .. rfa.values.len() {
       let v = rfa.values[i];
        let d = rfa.prev_value - v;
        if d > 0.0 {
            s += d
        }
        rfa.prev_value = v;
    }

    s
}

fn rollup_zscore_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // See https://about.gitlab.com/blog/2019/07/23/anomaly-detection-using-prometheus/#using-z-score-for-anomaly-detection
    let scrape_interval = rollup_scrape_interval(rfa);
    let lag = rollup_lag(rfa);
    if scrape_interval.is_nan() || lag.is_nan() || lag > scrape_interval {
        return NAN;
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
        return NAN;
    }
    return values[0];
}

fn rollup_default(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.len() == 0 {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
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
        return NAN;
    }
    return *values.last().unwrap() as f64;
}

fn rollup_distinct(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values;
    if values.len() == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
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
            return NAN;
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
