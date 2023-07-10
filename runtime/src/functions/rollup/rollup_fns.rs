use std::cell::RefCell;
use std::default::Default;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;

use lib::{get_float64s, is_stale_nan};
use metricsql::ast::{Expr, FunctionExpr};
use metricsql::functions::{can_adjust_window, RollupFunction};
use metricsql::prelude::BuiltinFunction;

use crate::eval::validate_max_points_per_timeseries;
use crate::functions::rollup::types::RollupHandlerFactory;
use crate::functions::rollup::{
    RollupFn, RollupFunc, RollupFuncArg, RollupHandler, RollupHandlerEnum,
};
use crate::functions::types::{get_scalar_param_value, get_string_param_value};
use crate::functions::{mode_no_nans, quantile, quantiles};
use crate::get_timestamps;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{get_timeseries, Timestamp};
use crate::QueryValue;

use super::timeseries_map::TimeseriesMap;

// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/app/vmselect/promql/rollup.go

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

/// The maximum interval without previous rows.
pub const MAX_SILENCE_INTERVAL: i64 = 5 * 60 * 1000;

pub(crate) fn get_rollup_fn(f: &RollupFunction) -> RuntimeResult<RollupFunc> {
    use RollupFunction::*;

    let res = match f {
        AbsentOverTime => rollup_absent,
        AscentOverTime => rollup_ascent_over_time,
        AvgOverTime => rollup_avg,
        Changes => rollup_changes,
        CountOverTime => rollup_count,
        DecreasesOverTime => ROLLUP_DECREASES,
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
        _ => {
            return Err(RuntimeError::General(format!(
                "{} is not an rollup function",
                &f.name()
            )))
        }
    };

    Ok(res)
}

fn get_rollup_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
    match RollupFunction::from_str(name) {
        Err(_) => Err(RuntimeError::UnknownFunction(name.to_string())),
        Ok(func) => Ok(func),
    }
}

pub(crate) fn rollup_func_requires_config(f: &RollupFunction) -> bool {
    use RollupFunction::*;

    matches!(
        f,
        PredictLinear
            | DurationOverTime
            | HoltWinters
            | ShareLeOverTime
            | ShareGtOverTime
            | CountLeOverTime
            | CountGtOverTime
            | CountEqOverTime
            | CountNeOverTime
            | HoeffdingBoundLower
            | HoeffdingBoundUpper
            | QuantilesOverTime
            | QuantileOverTime
    )
}

/// rollup_samples_scanned_per_call contains functions which scan lower number of samples
/// than is passed to the rollup func.
///
/// It is expected that the remaining rollupFuncs scan all the samples passed to them.
pub(super) fn rollup_samples_scanned_per_call(rf: &RollupFunction) -> usize {
    use RollupFunction::*;

    return match rf {
        AbsentOverTime => 1,
        CountOverTime => 1,
        DefaultRollup => 1,
        Delta => 2,
        DeltaPrometheus => 2,
        DerivFast => 2,
        FirstOverTime => 1,
        IDelta => 2,
        IDeriv => 2,
        Increase => 2,
        IncreasePrometheus => 2,
        IncreasePure => 2,
        IRate => 2,
        Lag => 1,
        LastOverTime => 1,
        Lifetime => 2,
        PresentOverTime => 1,
        Rate => 2,
        ScrapeInterval => 2,
        TFirstOverTime => 1,
        Timestamp => 1,
        TimestampWithName => 1,
        TLastOverTime => 1,
        _ => 0, // == num rows
    };
}

// Pre-allocated handlers for closure to save allocations at runtime
macro_rules! wrap_rollup_fn {
    ( $name: ident, $rf: expr ) => {
        const $name: RollupHandlerEnum = RollupHandlerEnum::Wrapped($rf);
    };
}

macro_rules! make_factory {
    ( $name: ident, $rf: expr ) => {
        #[inline]
        fn $name(_: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
            Ok(RollupHandlerEnum::wrap($rf))
        }
    };
}

macro_rules! fake_wrapper {
    ( $funcName: ident, $name: expr ) => {
        #[inline]
        fn $funcName(_: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
            Ok(RollupHandlerEnum::fake($name))
        }
    };
}

// pass through to existing functions

///////////
make_factory!(new_rollup_absent_over_time, rollup_absent);
make_factory!(new_rollup_aggr_over_time, rollup_fake);
make_factory!(new_rollup_ascent_over_time, rollup_ascent_over_time);
make_factory!(new_rollup_avg_over_time, rollup_avg);
make_factory!(new_rollup_changes, rollup_changes);
make_factory!(new_rollup_changes_prometheus, rollup_changes_prometheus);
make_factory!(new_rollup_count_over_time, rollup_count);
make_factory!(new_rollup_decreases_over_time, ROLLUP_DECREASES);
make_factory!(new_rollup_default, rollup_default);
make_factory!(new_rollup_delta, rollup_delta);
make_factory!(new_rollup_delta_prometheus, rollup_delta_prometheus);
make_factory!(new_rollup_deriv, rollup_deriv_slow);
make_factory!(new_rollup_deriv_fast, rollup_deriv_fast);
make_factory!(new_rollup_descent_over_time, rollup_descent_over_time);
make_factory!(new_rollup_distinct_over_time, rollup_distinct);
make_factory!(new_rollup_first_over_time, rollup_first);
make_factory!(new_rollup_geomean_over_time, rollup_geomean);
make_factory!(new_rollup_histogram_over_time, rollup_histogram);
make_factory!(new_rollup_idelta, rollup_idelta);
make_factory!(new_rollup_ideriv, rollup_ideriv);
make_factory!(new_rollup_increase, rollup_delta);
make_factory!(new_rollup_increase_pure, rollup_increase_pure);
make_factory!(new_rollup_increases_over_time, rollup_increases);
make_factory!(new_rollup_integrate, rollup_integrate);
make_factory!(new_rollup_irate, rollup_ideriv);
make_factory!(new_rollup_lag, rollup_lag);
make_factory!(new_rollup_last_over_time, rollup_last);
make_factory!(new_rollup_lifetime, rollup_lifetime);
make_factory!(new_rollup_mad_over_time, rollup_mad);
make_factory!(new_rollup_max_over_time, rollup_max);
make_factory!(new_rollup_min_over_time, rollup_min);
make_factory!(new_rollup_median_over_time, rollup_median);
make_factory!(new_rollup_mode_over_time, rollup_mode_over_time);
make_factory!(new_rollup_present_over_time, rollup_present);
make_factory!(new_rollup_range_over_time, rollup_range);
make_factory!(new_rollup_rate, rollup_deriv_fast);
make_factory!(new_rollup_rate_over_sum, rollup_rate_over_sum);
make_factory!(new_rollup_resets, rollup_resets);
make_factory!(new_rollup_scrape_interval, rollup_scrape_interval);
make_factory!(new_rollup_stale_samples_over_time, rollup_stale_samples);
make_factory!(new_rollup_stddev_over_time, rollup_stddev);
make_factory!(new_rollup_stdvar_over_time, rollup_stdvar);
make_factory!(new_rollup_sum_over_time, rollup_sum);
make_factory!(new_rollup_sum2_over_time, rollup_sum2);
make_factory!(new_rollup_tfirst_over_time, rollup_tfirst);
make_factory!(new_rollup_timestamp, rollup_tlast);
make_factory!(new_rollup_timestamp_with_name, rollup_tlast);
make_factory!(new_rollup_tlast_change_over_time, rollup_tlast_change);
make_factory!(new_rollup_tlast_over_time, rollup_tlast);
make_factory!(new_rollup_tmax_over_time, rollup_tmax);
make_factory!(new_rollup_tmin_over_time, rollup_tmin);
make_factory!(new_rollup_zscore_over_time, rollup_zscore_over_time);

fake_wrapper!(new_rollup, "rollup");
fake_wrapper!(new_rollup_candlestick, "rollup_candlestick");
fake_wrapper!(new_rollup_rollup_delta, "rollup_delta");
fake_wrapper!(new_rollup_rollup_deriv, "rollup_deriv");
fake_wrapper!(new_rollup_rollup_increase, "rollup_increase"); // + rollupFuncsRemoveCounterResets
fake_wrapper!(new_rollup_rollup_rate, "rollup_rate"); // + rollupFuncsRemoveCounterResets
fake_wrapper!(new_rollup_scrape_interval_fake, "rollup_scrape_interval");

pub(crate) fn get_rollup_function_factory(func: RollupFunction) -> RollupHandlerFactory {
    use RollupFunction::*;
    let imp = match func {
        AbsentOverTime => new_rollup_absent_over_time,
        AggrOverTime => new_rollup_aggr_over_time,
        AscentOverTime => new_rollup_ascent_over_time,
        AvgOverTime => new_rollup_avg_over_time,
        Changes => new_rollup_changes,
        ChangesPrometheus => new_rollup_changes_prometheus,
        CountEqOverTime => new_rollup_count_eq,
        CountGtOverTime => new_rollup_count_gt,
        CountLeOverTime => new_rollup_count_le,
        CountNeOverTime => new_rollup_count_ne,
        CountOverTime => new_rollup_count_over_time,
        DecreasesOverTime => new_rollup_decreases_over_time,
        DefaultRollup => new_rollup_default,
        Delta => new_rollup_delta,
        DeltaPrometheus => new_rollup_delta_prometheus,
        Deriv => new_rollup_deriv,
        DerivFast => new_rollup_deriv_fast,
        DescentOverTime => new_rollup_descent_over_time,
        DistinctOverTime => new_rollup_distinct_over_time,
        DurationOverTime => new_rollup_duration_over_time,
        FirstOverTime => new_rollup_first_over_time,
        GeomeanOverTime => new_rollup_geomean_over_time,
        HistogramOverTime => new_rollup_histogram_over_time,
        HoeffdingBoundLower => new_rollup_hoeffding_bound_lower,
        HoeffdingBoundUpper => new_rollup_hoeffding_bound_upper,
        HoltWinters => new_rollup_holt_winters,
        IDelta => new_rollup_idelta,
        IDeriv => new_rollup_ideriv,
        Increase => new_rollup_increase,
        IncreasePrometheus => new_rollup_delta_prometheus,
        IncreasePure => new_rollup_increase_pure,
        IncreasesOverTime => new_rollup_increases_over_time,
        Integrate => new_rollup_integrate,
        IRate => new_rollup_irate,
        Lag => new_rollup_lag,
        LastOverTime => new_rollup_last_over_time,
        Lifetime => new_rollup_lifetime,
        MadOverTime => new_rollup_mad_over_time,
        MaxOverTime => new_rollup_max_over_time,
        MedianOverTime => new_rollup_median_over_time,
        MinOverTime => new_rollup_min_over_time,
        ModeOverTime => new_rollup_mode_over_time,
        PredictLinear => new_rollup_predict_linear,
        PresentOverTime => new_rollup_present_over_time,
        QuantileOverTime => new_rollup_quantile,
        QuantilesOverTime => new_rollup_quantiles,
        RangeOverTime => new_rollup_range_over_time,
        Rate => new_rollup_rate,
        RateOverSum => new_rollup_rate_over_sum,
        Resets => new_rollup_resets,
        Rollup => new_rollup,
        RollupCandlestick => new_rollup_candlestick,
        RollupDelta => new_rollup_delta,
        RollupDeriv => new_rollup_deriv,
        RollupIncrease => new_rollup_increase,
        RollupRate => new_rollup_rate,
        RollupScrapeInterval => new_rollup_scrape_interval_fake,
        ScrapeInterval => new_rollup_scrape_interval,
        ShareGtOverTime => new_rollup_share_gt,
        ShareLeOverTime => new_rollup_share_le,
        StaleSamplesOverTime => new_rollup_stale_samples_over_time,
        StddevOverTime => new_rollup_stddev_over_time,
        StdvarOverTime => new_rollup_stdvar_over_time,
        SumOverTime => new_rollup_sum_over_time,
        Sum2OverTime => new_rollup_sum2_over_time,
        TFirstOverTime => new_rollup_tfirst_over_time,
        Timestamp => new_rollup_timestamp,
        TimestampWithName => new_rollup_timestamp_with_name,
        TLastChangeOverTime => new_rollup_tlast_change_over_time,
        TLastOverTime => new_rollup_tlast_over_time,
        TMaxOverTime => new_rollup_tmax_over_time,
        TMinOverTime => new_rollup_tmin_over_time,
        ZScoreOverTime => new_rollup_zscore_over_time,
    };

    return imp;
}

#[inline]
fn removes_counter_resets(f: RollupFunction) -> bool {
    use RollupFunction::*;
    matches!(
        f,
        Increase | IncreasePrometheus | IncreasePure | IRate | RollupIncrease | RollupRate
    )
}

pub(crate) fn rollup_func_keeps_metric_name(name: &str) -> bool {
    match RollupFunction::from_str(name) {
        Err(_) => false,
        Ok(func) => func.keep_metric_name(),
    }
}

// todo: use in optimize so its cached in the ast
pub(crate) fn get_rollup_aggr_funcs(expr: &Expr) -> RuntimeResult<Vec<RollupFunction>> {
    fn raise_err(expr: &Expr) -> RuntimeResult<Vec<RollupFunction>> {
        let msg = format!(
            "BUG: unexpected expression; want FunctionExpr; got {}; value: {}",
            expr.variant_name(),
            expr
        );
        Err(RuntimeError::from(msg))
    }

    match expr {
        Expr::Aggregation(afe) => {
            // This is for incremental aggregate function case:
            //
            //     sum(aggr_over_time(...))
            // See aggr_incremental.rs for details.
            let _expr = &afe.args[0];
            match _expr.deref() {
                Expr::Function(f) => get_rollup_aggr_funcs_impl(f),
                _ => raise_err(_expr),
            }
        }
        Expr::Function(fe) => get_rollup_aggr_funcs_impl(fe),
        _ => raise_err(expr),
    }
}

fn get_rollup_aggr_funcs_impl(fe: &FunctionExpr) -> RuntimeResult<Vec<RollupFunction>> {
    let is_aggr_over_time = match fe.function {
        BuiltinFunction::Rollup(rf) => rf == RollupFunction::AggrOverTime,
        _ => false,
    };

    if !is_aggr_over_time {
        let msg = format!(
            "BUG: unexpected function name: `{}`; want `aggr_over_time`",
            fe.name
        );
        return Err(RuntimeError::from(msg));
    }

    let arg_len = fe.args.len();
    if arg_len < 2 {
        let msg = format!(
            "unexpected number of args to aggr_over_time(); got {arg_len}; want at least 2"
        );
        return Err(RuntimeError::from(msg));
    }

    let mut aggr_funcs: Vec<RollupFunction> = Vec::with_capacity(1);
    for arg in fe.args.iter() {
        match arg.deref() {
            Expr::StringLiteral(name) => match get_rollup_func_by_name(&name) {
                Err(_) => {
                    let msg = format!("{name} cannot be used in `aggr_over_time` function; expecting quoted aggregate function name");
                    return Err(RuntimeError::General(msg));
                }
                Ok(rf) => aggr_funcs.push(rf),
            },
            _ => {
                let msg = format!(
                    "{arg} cannot be passed here; expecting quoted aggregate function name",
                );
                return Err(RuntimeError::General(msg));
            }
        }
    }

    Ok(aggr_funcs)
}

fn get_rollup_tag(expr: &Expr) -> RuntimeResult<Option<&String>> {
    return if let Expr::Function(fe) = expr {
        if fe.args.len() < 2 {
            return Ok(None);
        }
        if fe.args.len() != 2 {
            let msg = format!(
                "unexpected number of args for rollup function {}; got {:?}; want 2",
                fe.name, fe.args
            );
            return Err(RuntimeError::General(msg));
        }
        let arg = &fe.args[1];
        if let Expr::StringLiteral(se) = arg {
            if se.is_empty() {
                return Err(RuntimeError::ArgumentError(
                    "unexpected empty rollup tag value".to_string(),
                ));
            }
            Ok(Some(se))
        } else {
            Err(RuntimeError::ArgumentError(format!(
                "unexpected rollup tag value {arg}; wanted min, max or avg",
            )))
        }
    } else {
        let msg = format!("BUG: unexpected expression; want FunctionExpr; got {expr};");
        Err(RuntimeError::ArgumentError(msg))
    };
}

pub(crate) type PreFunction = fn(&mut [f64], &[i64]) -> ();

type FloatComparisonFunction = fn(left: f64, right: f64) -> bool;

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
    for (value, ts) in values.iter_mut().zip(timestamps.iter()) {
        let ts_secs = (ts / 1000) as f64;
        *value = ts_secs - ts_secs_prev;
        ts_secs_prev = ts_secs;
    }

    if values.len() > 1 {
        // Overwrite the first NaN interval with the second interval,
        // So min, max and avg rollup could be calculated properly,
        // since they don't expect to receive NaNs.
        values[0] = values[1]
    }
}

wrap_rollup_fn!(FN_OPEN, rollup_open);
wrap_rollup_fn!(FN_CLOSE, rollup_close);
wrap_rollup_fn!(FN_MIN, rollup_min);
wrap_rollup_fn!(FN_MAX, rollup_max);
wrap_rollup_fn!(FN_AVG, rollup_avg);
wrap_rollup_fn!(FN_LOW, rollup_low);
wrap_rollup_fn!(FN_HIGH, rollup_high);
wrap_rollup_fn!(FN_FAKE, rollup_fake);

pub(crate) fn get_rollup_configs<'a>(
    func: &RollupFunction,
    rf: &'a RollupHandlerEnum,
    expr: &Expr,
    start: Timestamp,
    end: Timestamp,
    step: i64,
    window: i64,
    max_points_per_series: usize,
    min_staleness_interval: usize,
    lookback_delta: i64,
    shared_timestamps: &Arc<Vec<i64>>,
) -> RuntimeResult<(Vec<RollupConfig>, Vec<PreFunction>)> {
    // todo: use tinyvec
    let mut pre_funcs: Vec<PreFunction> = Vec::with_capacity(3);

    if func.should_remove_counter_resets() {
        pre_funcs.push(remove_counter_resets_pre_func);
    }

    let may_adjust_window = can_adjust_window(func);
    let is_default_rollup = *func == RollupFunction::DefaultRollup;
    let samples_scanned_per_call = rollup_samples_scanned_per_call(func);

    let template = RollupConfig {
        tag_value: String::from(""),
        handler: FN_FAKE.clone(),
        start,
        end,
        step,
        window,
        may_adjust_window,
        lookback_delta,
        timestamps: Arc::clone(&shared_timestamps),
        is_default_rollup,
        max_points_per_series,
        min_staleness_interval,
        samples_scanned_per_call,
    };

    let new_rollup_config = |rf: &RollupHandlerEnum, tag_value: &str| -> RollupConfig {
        template.clone_with_fn(rf, tag_value)
    };

    let append_rollup_configs = |dst: &mut Vec<RollupConfig>, expr: &Expr| -> RuntimeResult<()> {
        let tag = get_rollup_tag(expr)?;
        if let Some(tag) = tag {
            match tag.as_str() {
                "min" => dst.push(new_rollup_config(&FN_MIN, "min")),
                "max" => dst.push(new_rollup_config(&FN_MAX, "max")),
                "avg" => dst.push(new_rollup_config(&FN_AVG, "avg")),
                _ => {
                    let msg = format!(
                        "unexpected rollup tag value {}; wanted min, max or avg",
                        tag
                    );
                    return Err(RuntimeError::ArgumentError(msg));
                }
            }
        } else {
            dst.push(new_rollup_config(&FN_MIN, "min"));
            dst.push(new_rollup_config(&FN_MAX, "max"));
            dst.push(new_rollup_config(&FN_AVG, "avg"));
        }
        Ok(())
    };

    // todo: tinyvec
    let mut rcs: Vec<RollupConfig> = Vec::with_capacity(1);
    match func {
        RollupFunction::Rollup => {
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupRate | RollupFunction::RollupDeriv => {
            pre_funcs.push(deriv_values);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupIncrease | RollupFunction::RollupDelta => {
            pre_funcs.push(delta_values_pre_func);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::RollupCandlestick => {
            let tag = get_rollup_tag(expr)?;
            if let Some(tag) = tag {
                match tag.as_str() {
                    "open" => rcs.push(new_rollup_config(&FN_OPEN, "open")),
                    "close" => rcs.push(new_rollup_config(&FN_CLOSE, "close")),
                    "low" => rcs.push(new_rollup_config(&FN_LOW, "low")),
                    "high" => rcs.push(new_rollup_config(&FN_HIGH, "high")),
                    _ => {
                        let msg = format!(
                            "unexpected rollup tag value {}; wanted open, close, low or high",
                            tag
                        );
                        return Err(RuntimeError::ArgumentError(msg));
                    }
                }
            } else {
                rcs.push(new_rollup_config(&FN_OPEN, "open"));
                rcs.push(new_rollup_config(&FN_CLOSE, "close"));
                rcs.push(new_rollup_config(&FN_LOW, "low"));
                rcs.push(new_rollup_config(&FN_HIGH, "high"));
            }
        }
        RollupFunction::RollupScrapeInterval => {
            pre_funcs.push(calc_sample_intervals_pre_fn);
            append_rollup_configs(&mut rcs, expr)?;
        }
        RollupFunction::AggrOverTime => {
            match get_rollup_aggr_funcs(expr) {
                Err(_) => {
                    return Err(RuntimeError::ArgumentError(format!(
                        "invalid args to {expr}"
                    )))
                }
                Ok(funcs) => {
                    for rf in funcs {
                        if removes_counter_resets(rf) {
                            // There is no need to save the previous pre_func, since it is either empty or the same.
                            pre_funcs.clear();
                            pre_funcs.push(remove_counter_resets_pre_func);
                        }
                        let rollup_fn = get_rollup_fn(&rf)?;
                        let handler = RollupHandlerEnum::wrap(rollup_fn);
                        let clone = template.clone_with_fn(&handler, &rf.name());
                        rcs.push(clone);
                    }
                }
            }
        }
        _ => {
            rcs.push(new_rollup_config(rf, ""));
        }
    }

    Ok((rcs, pre_funcs))
}

#[derive(Clone)]
pub(crate) struct RollupConfig {
    /// This tag value must be added to "rollup" tag if non-empty.
    pub tag_value: String,
    pub handler: RollupHandlerEnum,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub window: i64,

    /// Whether window may be adjusted to 2 x interval between data points.
    /// This is needed for functions which have dt in the denominator
    /// such as rate, deriv, etc.
    /// Without the adjustment their value would jump in unexpected directions
    /// when using window smaller than 2 x scrape_interval.
    pub may_adjust_window: bool,

    pub timestamps: Arc<Vec<i64>>,

    /// lookback_delta is the analog to `-query.lookback-delta` from Prometheus world.
    pub lookback_delta: i64,

    /// Whether default_rollup is used.
    pub is_default_rollup: bool,

    /// The maximum number of points which can be generated per each series.
    pub max_points_per_series: usize,

    /// The minimum interval for staleness calculations. This could be useful for removing gaps on
    /// graphs generated from time series with irregular intervals between samples.
    pub min_staleness_interval: usize,

    /// The estimated number of samples scanned per Func call.
    ///
    /// If zero, then it is considered that Func scans all the samples passed to it.
    pub samples_scanned_per_call: usize,
}

impl Default for RollupConfig {
    fn default() -> Self {
        Self {
            tag_value: "".to_string(),
            handler: RollupHandlerEnum::Fake("uninitialized"),
            start: 0,
            end: 0,
            step: 0,
            window: 0,
            may_adjust_window: false,
            timestamps: Arc::new(vec![]),
            lookback_delta: 0,
            is_default_rollup: false,
            max_points_per_series: 0,
            min_staleness_interval: 0,
            samples_scanned_per_call: 0,
        }
    }
}

impl RollupConfig {
    fn clone_with_fn(&self, rollup_fn: &RollupHandlerEnum, tag_value: &str) -> Self {
        return RollupConfig {
            tag_value: tag_value.to_string(), // should this be Arc ??
            handler: rollup_fn.clone(),
            start: self.start,
            end: self.end,
            step: self.step,
            window: self.window,
            may_adjust_window: self.may_adjust_window,
            lookback_delta: self.lookback_delta,
            timestamps: Arc::clone(&self.timestamps),
            is_default_rollup: self.is_default_rollup,
            max_points_per_series: self.max_points_per_series,
            min_staleness_interval: self.min_staleness_interval,
            samples_scanned_per_call: self.samples_scanned_per_call,
        };
    }

    // mostly for testing
    pub(crate) fn get_timestamps(&mut self) -> RuntimeResult<Arc<Vec<i64>>> {
        self.ensure_timestamps()?;
        Ok(Arc::clone(&self.timestamps))
    }

    pub(crate) fn ensure_timestamps(&mut self) -> RuntimeResult<()> {
        if self.timestamps.len() == 0 {
            let ts = get_timestamps(self.start, self.end, self.step, self.max_points_per_series)?;
            self.timestamps = Arc::new(ts);
        }
        Ok(())
    }

    /// calculates rollup for the given timestamps and values, appends
    /// them to dst_values and returns results.
    ///
    /// rc.timestamps are used as timestamps for dst_values.
    ///
    /// timestamps must cover time range [rc.start - rc.window - MAX_SILENCE_INTERVAL ... rc.end].
    pub(crate) fn exec(
        &self,
        dst_values: &mut Vec<f64>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        self.do_internal(dst_values, None, values, timestamps)
    }

    /// calculates rollup for the given timestamps and values and puts them to tsm.
    /// returns the number of samples scanned
    pub(crate) fn do_timeseries_map(
        &self,
        tsm: &Rc<RefCell<TimeseriesMap>>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        let mut ts = get_timeseries();
        self.do_internal(&mut ts.values, Some(tsm), values, timestamps)
    }

    fn do_internal(
        &self,
        dst_values: &mut Vec<f64>,
        tsm: Option<&Rc<RefCell<TimeseriesMap>>>,
        values: &[f64],
        timestamps: &[Timestamp],
    ) -> RuntimeResult<u64> {
        // Sanity checks.
        self.validate()?;

        // Extend dst_values in order to remove allocations below.
        dst_values.reserve(self.timestamps.len());

        let scrape_interval = get_scrape_interval(&timestamps);
        let mut max_prev_interval = get_max_prev_interval(scrape_interval);
        if self.lookback_delta > 0 && max_prev_interval > self.lookback_delta {
            max_prev_interval = self.lookback_delta
        }
        if self.min_staleness_interval > 0 {
            let msi = self.min_staleness_interval as i64;
            if msi > 0 && max_prev_interval < msi {
                max_prev_interval = msi
            }
        }
        let mut window = self.window;
        if window <= 0 {
            window = self.step;
            if self.may_adjust_window && window < max_prev_interval {
                // Adjust lookbehind window only if it isn't set explicitly, e.g. rate(foo).
                // In the case of missing lookbehind window it should be adjusted in order to return non-empty graph
                // when the window doesn't cover at least two raw samples (this is what most users expect).
                //
                // If the user explicitly sets the lookbehind window to some fixed value, e.g. rate(foo[1s]),
                // then it is expected he knows what he is doing. Do not adjust the lookbehind window then.
                //
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3483
                window = max_prev_interval
            }
            if self.is_default_rollup && self.lookback_delta > 0 && window > self.lookback_delta {
                // Implicit window exceeds -search.maxStalenessInterval, so limit it to
                // -search.maxStalenessInterval
                // according to https://github.com/VictoriaMetrics/VictoriaMetrics/issues/784
                window = self.lookback_delta
            }
        }

        let mut rfa = RollupFuncArg::default();
        rfa.idx = 0;
        rfa.window = window;
        rfa.tsm = if let Some(t) = tsm {
            Some(Rc::clone(t))
        } else {
            None
        };

        let mut i = 0;
        let mut j = 0;
        let mut ni = 0;
        let mut nj = 0;

        let mut samples_scanned = values.len() as u64;
        let samples_scanned_per_call = self.samples_scanned_per_call as u64;

        for t_end in self.timestamps.iter() {
            let t_start = *t_end - window;
            ni = seek_first_timestamp_idx_after(&timestamps[i..], t_start, ni);
            i += ni;
            if j < i {
                j = i;
            }

            nj = seek_first_timestamp_idx_after(&timestamps[j..], *t_end, nj);
            j += nj;

            rfa.prev_value = NAN;
            rfa.prev_timestamp = t_start - max_prev_interval;
            if i > 0 && i < timestamps.len() {
                let prev_ts = timestamps[i - 1];
                if prev_ts > rfa.prev_timestamp {
                    rfa.prev_value = values[i - 1];
                    rfa.prev_timestamp = prev_ts;
                }
            }

            rfa.values.clear();
            rfa.timestamps.clear();
            rfa.values.extend_from_slice(&values[i..j]);
            rfa.timestamps.extend_from_slice(&timestamps[i..j]);

            rfa.real_prev_value = if i > 0 { values[i - 1] } else { NAN };
            rfa.real_next_value = if j < values.len() { values[j] } else { NAN };

            rfa.curr_timestamp = *t_end;
            let value = (self.handler).eval(&mut rfa);
            rfa.idx += 1;

            if samples_scanned_per_call > 0 {
                samples_scanned += samples_scanned_per_call
            } else {
                samples_scanned += rfa.values.len() as u64;
            }

            dst_values.push(value);
        }

        Ok(samples_scanned)
    }

    fn validate(&self) -> RuntimeResult<()> {
        // Sanity checks.
        if self.step <= 0 {
            let msg = format!("BUG: Step must be bigger than 0; got {}", self.step);
            return Err(RuntimeError::from(msg));
        }
        if self.start > self.end {
            let msg = format!(
                "BUG: start cannot exceed end; got {} vs {}",
                self.start, self.end
            );
            return Err(RuntimeError::from(msg));
        }
        if self.window < 0 {
            let msg = format!("BUG: Window must be non-negative; got {}", self.window);
            return Err(RuntimeError::from(msg));
        }
        match validate_max_points_per_timeseries(
            self.start,
            self.end,
            self.step,
            self.max_points_per_series,
        ) {
            Err(err) => {
                let msg = format!(
                    "BUG: {:?}; this must be validated before the call to rollupConfig.exec",
                    err
                );
                return Err(RuntimeError::from(msg));
            }
            _ => Ok(()),
        }
    }
}

fn seek_first_timestamp_idx_after(
    timestamps: &[Timestamp],
    seek_timestamp: Timestamp,
    n_hint: usize,
) -> usize {
    let mut timestamps = timestamps;
    let count = timestamps.len();

    if count == 0 || timestamps[0] > seek_timestamp {
        return 0;
    }
    let mut start_idx = if n_hint >= 2 { n_hint - 2 } else { 0 };
    if start_idx >= count {
        start_idx = count - 1
    }

    let mut end_idx = n_hint + 2;
    if end_idx > count {
        end_idx = count
    }
    if start_idx > 0 && timestamps[start_idx] <= seek_timestamp {
        timestamps = &timestamps[start_idx..];
        end_idx -= start_idx
    } else {
        start_idx = 0
    }
    if end_idx < timestamps.len() && timestamps[end_idx] > seek_timestamp {
        timestamps = &timestamps[0..end_idx];
    }
    if timestamps.len() < 16 {
        // Fast path: the number of timestamps to search is small, so scan them all.
        for (i, timestamp) in timestamps.iter().enumerate() {
            if *timestamp > seek_timestamp {
                return start_idx + i;
            }
        }
        return start_idx + timestamps.len();
    }
    // Slow path: too big timestamps.len(), so use binary search.
    let requested = seek_timestamp + 1;
    match timestamps.binary_search(&requested) {
        Ok(pos) => start_idx + pos,
        Err(suggested) => start_idx + suggested,
    }
}

fn get_scrape_interval(timestamps: &[Timestamp]) -> i64 {
    if timestamps.len() < 2 {
        return MAX_SILENCE_INTERVAL;
    }

    // Estimate scrape interval as 0.6 quantile for the first 20 intervals.
    let mut ts_prev = timestamps[0];
    let timestamps = &timestamps[1..];
    let len = timestamps.len().clamp(0, 20);

    let mut intervals: [f64; 20] = [0_f64; 20];
    for (interval, ts) in intervals.iter_mut().zip(timestamps.iter()) {
        *interval = (ts - ts_prev) as f64;
        ts_prev = *ts
    }
    let scrape_interval = quantile(0.6, &intervals[0..len]) as i64;
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

pub(super) fn remove_counter_resets(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
    if values.is_empty() {
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
                correction += prev_value - *v;
            } else {
                correction += prev_value;
            }
        }
        prev_value = *v;
        *v += correction;
    }
}

pub(super) fn delta_values(values: &mut [f64]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
    if values.is_empty() {
        return;
    }

    let mut prev_delta: f64 = 0.0;
    let mut prev_value = values[0];

    for i in 1..values.len() {
        let v = values[i];
        prev_delta = v - prev_value;
        values[i - 1] = prev_delta;
        prev_value = v;
    }

    values[values.len() - 1] = prev_delta
}

pub(super) fn deriv_values(values: &mut [f64], timestamps: &[Timestamp]) {
    // There is no need in handling NaNs here, since they are impossible
    // on values from storage.
    if values.is_empty() {
        return;
    }
    let mut prev_deriv: f64 = 0.0;
    let mut prev_value = values[0];
    let mut prev_ts = timestamps[0];

    let mut j: usize = 0;
    for i in 1..values.len() {
        let v = values[i];
        let ts = timestamps[i];
        if ts == prev_ts {
            // Use the previous value for duplicate timestamps.
            values[j] = prev_deriv;
            j += 1;
            continue;
        }
        let dt = (ts - prev_ts) as f64 / 1e3_f64;
        prev_deriv = (v - prev_value) / dt;
        values[j] = prev_deriv;
        prev_value = v;
        prev_ts = ts;
        j += 1;
    }

    values[values.len() - 1] = prev_deriv
}

fn new_rollup_holt_winters(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    // unwrap is sound since arguments are checked before this is called
    let sf = get_scalar_param_value(args, 1, "holt_winters", "sf")?;
    let tf = get_scalar_param_value(args, 2, "holt_winters", "tf")?;

    Ok(RollupHandlerEnum::General(Box::new(
        move |rfa: &mut RollupFuncArg| -> f64 { holt_winters_internal(rfa, sf, tf) },
    )))
}

fn holt_winters_internal(rfa: &mut RollupFuncArg, sf: f64, tf: f64) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return rfa.prev_value;
    }

    if sf <= 0.0 || sf >= 1.0 {
        return NAN;
    }

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
    for v in rfa.values[ofs..].iter() {
        let s1 = sf * v + (1.0 - sf) * (s0 + b0);
        let b1 = tf * (s1 - s0) + (1.0 - tf) * b0;
        s0 = s1;
        b0 = b1
    }

    s0
}

fn new_rollup_predict_linear(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let secs = get_scalar_param_value(args, 1, "predict_linear", "secs")?;

    let res = RollupHandlerEnum::General(Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (v, k) = linear_regression(&rfa.values, &rfa.timestamps, rfa.curr_timestamp);
        if v.is_nan() {
            return NAN;
        }
        return v + k * secs;
    }));

    Ok(res)
}

pub(crate) fn linear_regression(
    values: &[f64],
    timestamps: &[i64],
    intercept_time: i64,
) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (NAN, NAN);
    }
    if are_const_values(values) {
        return (values[0], 0.0);
    }

    // See https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    let mut v_sum: f64 = 0.0;
    let mut t_sum: f64 = 0.0;
    let mut tv_sum: f64 = 0.0;
    let mut tt_sum: f64 = 0.0;

    for (ts, v) in timestamps.iter().zip(values.iter()) {
        let dt = (ts - intercept_time) as f64 / 1e3_f64;
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

fn are_const_values(values: &[f64]) -> bool {
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

fn new_rollup_duration_over_time(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let max_interval = get_scalar_param_value(args, 1, "duration_over_time", "max_interval")?;

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.timestamps.is_empty() {
            return NAN;
        }
        let mut t_prev = rfa.timestamps[0];
        let mut d_sum: i64 = 0;
        let d_max = (max_interval * 1000_f64) as i64;
        for t in rfa.timestamps.iter() {
            let d = t - t_prev;
            if d <= d_max {
                d_sum += d;
            }
            t_prev = *t
        }

        d_sum as f64 / 1000_f64
    });

    Ok(RollupHandlerEnum::General(f))
}

fn new_rollup_share_le(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_share_filter(args, "share_le_over_time", "le", |x, v| x <= v)
}

fn new_rollup_share_gt(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_share_filter(args, "share_gt_over_time", "gt", |x, v| x > v)
}

fn new_rollup_share_filter(
    args: &Vec<QueryValue>,
    func_name: &str,
    param_name: &str,
    count_filter: FloatComparisonFunction,
) -> RuntimeResult<RollupHandlerEnum> {
    let rf = new_rollup_count_filter(args, func_name, param_name, count_filter)?;
    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let n = rf.eval(rfa);
        return n / rfa.values.len() as f64;
    });

    Ok(RollupHandlerEnum::General(f))
}

fn new_rollup_count_le(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_count_filter(args, "count_le_over_time", "le", |x, v| x <= v)
}

fn new_rollup_count_gt(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_count_filter(args, "count_gt_over_time", "gt", |x, v| x > v)
}

fn new_rollup_count_eq(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_count_filter(args, "count_eq_over_time", "eq", |x, v| x == v)
}

fn new_rollup_count_ne(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    new_rollup_count_filter(args, "count_ne_over_time", "ne", |x, v| x != v)
}

fn new_rollup_count_filter(
    args: &Vec<QueryValue>,
    func_name: &str,
    param_name: &str,
    count_predicate: FloatComparisonFunction,
) -> RuntimeResult<RollupHandlerEnum> {
    let limit = get_scalar_param_value(args, 1, func_name, param_name)?;

    let handler = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.values.is_empty() {
            return NAN;
        }

        let mut n = 0;
        for v in rfa.values.iter() {
            if count_predicate(*v, limit) {
                n += 1;
            }
        }

        n as f64
    });

    Ok(RollupHandlerEnum::General(handler))
}

fn new_rollup_hoeffding_bound_lower(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi = get_scalar_param_value(args, 0, "hoeffding_bound_lower", "phi")?;

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(rfa, phi);
        avg - bound
    });

    Ok(RollupHandlerEnum::General(f))
}

fn new_rollup_hoeffding_bound_upper(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi = get_scalar_param_value(args, 0, "hoeffding_bound_upper", "phi")?;

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = rollup_hoeffding_bound_internal(rfa, phi);
        return avg + bound;
    });

    Ok(RollupHandlerEnum::General(f))
}

fn rollup_hoeffding_bound_internal(rfa: &mut RollupFuncArg, phi: f64) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return (NAN, NAN);
    }
    if rfa.values.len() == 1 {
        return (0.0, rfa.values[0]);
    }
    // todo(perf): use online algorithm to calculate min, max, avg
    let v_max = rollup_max(rfa);
    let v_min = rollup_min(rfa);
    let v_avg = rollup_avg(rfa);
    let v_range = v_max - v_min;
    if v_range <= 0.0 {
        return (0.0, v_avg);
    }

    if phi >= 1.0 {
        return (INF, v_avg);
    }

    if phi <= 0.0 {
        return (0.0, v_avg);
    }
    // See https://en.wikipedia.org/wiki/Hoeffding%27s_inequality
    // and https://www.youtube.com/watch?v=6UwcqiNsZ8U&feature=youtu.be&t=1237

    // let bound = v_range * math.Sqrt(math.Log(1 / (1 - phi)) / (2 * values.len()));
    let bound = v_range * ((1.0 / (1.0 - phi)).ln() / (2 * rfa.values.len()) as f64).sqrt();
    return (bound, v_avg);
}

fn new_rollup_quantiles(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi_label = get_string_param_value(&args[0], "quantiles", "phi_label").unwrap();
    let cap = args.len() - 1;

    let mut phis = Vec::with_capacity(cap);
    // todo: smallvec ??
    let mut phi_labels: Vec<String> = Vec::with_capacity(cap);

    for i in 1..args.len() {
        // unwrap should be safe, since parameters types are checked before calling the function
        let v = get_scalar_param_value(args, i, "quantiles", "phi").unwrap();
        phis.push(v);
        phi_labels.push(format!("{}", v));
    }

    let f: Box<dyn RollupFn<Output = f64>> = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        if rfa.values.is_empty() {
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
        let tsm = rfa.tsm.as_ref().unwrap();
        let mut wrapped = tsm.borrow_mut();
        for (phi_str, quantile) in phi_labels.iter().zip(qs.iter()) {
            let ts = wrapped.get_or_create_timeseries(&phi_label, phi_str);
            ts.values[idx] = *quantile;
        }

        return NAN;
    });

    Ok(RollupHandlerEnum::General(f))
}

fn new_rollup_quantile(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi = get_scalar_param_value(args, 0, "quantile_over_time", "phi")?;

    let rf = Box::new(move |rfa: &mut RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        quantile(phi, &rfa.values)
    });

    Ok(RollupHandlerEnum::General(rf))
}

pub(super) fn rollup_histogram(rfa: &mut RollupFuncArg) -> f64 {
    let tsm = rfa.tsm.as_ref().unwrap();
    let mut map = tsm.borrow_mut();
    map.reset();
    for v in rfa.values.iter() {
        map.update(*v);
    }
    let idx = rfa.idx;
    let ranges: Vec<(String, u64)> = map
        .non_zero_buckets()
        .map(|b| (b.vm_range.to_string(), b.count))
        .collect();

    for (vm_range, count) in ranges {
        let ts = map.get_or_create_timeseries("vmrange", &vm_range);
        ts.values[idx] = count as f64;
    }

    return NAN;
}

pub(super) fn rollup_avg(rfa: &mut RollupFuncArg) -> f64 {
    // do not use `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation,
    // since it is slower and has no significant benefits in precision.

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    let sum: f64 = rfa.values.iter().sum();
    return sum / rfa.values.len() as f64;
}

pub(super) fn rollup_min(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    let mut min_value = rfa.values[0];
    for v in rfa.values.iter() {
        if *v < min_value {
            min_value = *v;
        }
    }

    min_value
}

pub(crate) fn rollup_mad(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup funcs.

    mad(&rfa.values)
}

pub(crate) fn mad(values: &[f64]) -> f64 {
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    let median = quantile(0.5, values);
    let mut ds = get_float64s(values.len());
    for v in values.iter() {
        ds.push((v - median).abs())
    }
    quantile(0.5, &ds)
}

pub(super) fn rollup_max(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    *rfa.values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

pub(super) fn rollup_median(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    quantile(0.5, &rfa.values)
}

pub(super) fn rollup_tmin(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.is_empty() {
        return NAN;
    }
    let mut min_value = values[0];
    let mut min_timestamp = rfa.timestamps[0];
    for (v, ts) in rfa.values.iter().zip(rfa.timestamps.iter()) {
        // Get the last timestamp for the minimum value as most users expect.
        if v <= &min_value {
            min_value = *v;
            min_timestamp = *ts;
        }
    }
    return min_timestamp as f64 / 1e3_f64;
}

pub(super) fn rollup_tmax(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    if rfa.values.is_empty() {
        return NAN;
    }

    let mut max_value = rfa.values[0];
    let mut max_timestamp = rfa.timestamps[0];

    for (v, ts) in rfa.values.iter().zip(rfa.timestamps.iter()) {
        // Get the last timestamp for the maximum value as most users expect.
        if *v >= max_value {
            max_value = *v;
            max_timestamp = *ts;
        }
    }

    return max_timestamp as f64 / 1e3_f64;
}

pub(super) fn rollup_tfirst(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_tlast(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        // do not take into account rfa.prev_timestamp, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    return timestamps[timestamps.len() - 1] as f64 / 1e3_f64;
}

pub(super) fn rollup_tlast_change(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }

    let last = rfa.values.len() - 1;
    let last_value = rfa.values[last];

    for i in (0..last).rev() {
        if rfa.values[i] != last_value {
            return rfa.timestamps[i + 1] as f64 / 1e3_f64;
        }
    }

    if rfa.prev_value.is_nan() || rfa.prev_value != last_value {
        return rfa.timestamps[0] as f64 / 1e3_f64;
    }
    return NAN;
}

pub(super) fn rollup_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }

    rfa.values.iter().fold(0.0, |r, x| r + *x)
}

pub(super) fn rollup_rate_over_sum(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let timestamps = &rfa.timestamps;
    if timestamps.len() == 0 {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        // Assume that the value didn't change since rfa.prev_value.
        return 0.0;
    }
    let sum: f64 = rfa.values.iter().sum();
    return sum / (rfa.window as f64 / 1e3_f64);
}

pub(super) fn rollup_range(rfa: &mut RollupFuncArg) -> f64 {
    // todo(perf): calc internally rather than calling rollup_max and rollup_min
    let max = rollup_max(rfa);
    let min = rollup_min(rfa);
    return max - min;
}

pub(super) fn rollup_sum2(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return rfa.prev_value * rfa.prev_value;
    }
    rfa.values.iter().fold(0.0, |r, x| r + (*x * *x))
}

pub(super) fn rollup_geomean(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let len = rfa.values.len();
    if len == 0 {
        return rfa.prev_value;
    }

    let p = rfa.values.iter().fold(1.0, |r, v| r * *v);
    return p.powf((1 / len) as f64);
}

pub(super) fn rollup_absent(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.values.is_empty() {
        return 1.0;
    }
    return NAN;
}

pub(super) fn rollup_present(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.len() > 0 {
        return 1.0;
    }
    return NAN;
}

pub(super) fn rollup_count(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return NAN;
    }
    return rfa.values.len() as f64;
}

pub(super) fn rollup_stale_samples(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.is_empty() {
        return NAN;
    }
    rfa.values.iter().filter(|v| is_stale_nan(**v)).count() as f64
}

pub(super) fn rollup_stddev(rfa: &mut RollupFuncArg) -> f64 {
    stddev(&rfa.values)
}

pub(super) fn rollup_stdvar(rfa: &mut RollupFuncArg) -> f64 {
    stdvar(&rfa.values)
}

pub(crate) fn stddev(values: &[f64]) -> f64 {
    let std_var = stdvar(values);
    return std_var.sqrt();
}

pub(crate) fn stdvar(values: &[f64]) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation

    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if values.is_empty() {
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
        if v.is_nan() {
            continue;
        }
        count += 1;
        let avg_new = avg + (*v - avg) / count as f64;
        q += (*v - avg) * (*v - avg_new);
        avg = avg_new
    }
    if count == 0 {
        return NAN;
    }
    return q / count as f64;
}

pub(super) fn rollup_increase_pure(rfa: &mut RollupFuncArg) -> f64 {
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
    if rfa.values.is_empty() {
        // Assume the counter didn't change since prev_value.
        return 0 as f64;
    }
    return rfa.values[count - 1] - rfa.prev_value;
}

pub(super) fn rollup_delta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    if rfa.prev_value.is_nan() {
        if values.is_empty() {
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

        let d = if rfa.values.len() > 1 {
            rfa.values[1] - rfa.values[0]
        } else if !rfa.real_next_value.is_nan() {
            rfa.real_next_value - values[0]
        } else {
            0.0
        };

        if rfa.values[0].abs() < 10.0 * (d.abs() + 1.0) {
            rfa.prev_value = 0.0;
        } else {
            rfa.prev_value = rfa.values[0];
            values = &values[1..]
        }
    }
    if values.is_empty() {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    return values[values.len() - 1] - rfa.prev_value;
}

pub(super) fn rollup_delta_prometheus(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_idelta(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.is_empty() {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let last_value = rfa.values[rfa.values.len() - 1];
    let values = &values[0..values.len() - 1];
    if values.is_empty() {
        let prev_value = rfa.prev_value;
        if prev_value.is_nan() {
            // Assume that the previous non-existing value was 0.
            return last_value;
        }
        return last_value - prev_value;
    }
    return last_value - values[values.len() - 1];
}

pub(super) fn rollup_deriv_slow(rfa: &mut RollupFuncArg) -> f64 {
    // Use linear regression like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/73
    let (_, k) = linear_regression(&rfa.values, &rfa.timestamps, rfa.curr_timestamp);
    k
}

pub(super) fn rollup_deriv_fast(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let timestamps = &rfa.timestamps;
    let mut prev_value = rfa.prev_value;
    let mut prev_timestamp = rfa.prev_timestamp;
    if prev_value.is_nan() {
        if values.is_empty() {
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
    } else if values.is_empty() {
        // Assume that the value didn't change on the given interval.
        return 0.0;
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];
    let dv = v_end - prev_value;
    let dt = (t_end - prev_timestamp) as f64 / 1e3_f64;
    return dv / dt;
}

pub(super) fn rollup_ideriv(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
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
        return (values[0] - rfa.prev_value)
            / ((timestamps[0] - rfa.prev_timestamp) as f64 / 1e3_f64);
    }
    let v_end = values[values.len() - 1];
    let t_end = timestamps[timestamps.len() - 1];

    let values = &values[0..count - 1];
    let mut timestamps = &timestamps[0..timestamps.len() - 1];

    // Skip data points with duplicate timestamps.
    while timestamps.len() > 0 && timestamps[timestamps.len() - 1] >= t_end {
        timestamps = &timestamps[0..timestamps.len() - 1];
    }
    count = timestamps.len();

    let t_start: i64;
    let v_start: f64;
    if count == 0 {
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

pub(super) fn rollup_lifetime(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_lag(rfa: &mut RollupFuncArg) -> f64 {
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

/// Calculate the average interval between data points.
pub(super) fn rollup_scrape_interval(rfa: &mut RollupFuncArg) -> f64 {
    let count = rfa.timestamps.len();
    if rfa.prev_value.is_nan() {
        if count < 2 {
            return NAN;
        }
        return ((rfa.timestamps[count - 1] - rfa.timestamps[0]) as f64 / 1e3_f64)
            / (count - 1) as f64;
    }
    if count == 0 {
        return NAN;
    }
    return ((rfa.timestamps[count - 1] - rfa.prev_timestamp) as f64 / 1e3_f64) / count as f64;
}

pub(super) fn rollup_changes_prometheus(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // do not take into account rfa.prev_value like Prometheus does.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1962
    if rfa.values.len() < 1 {
        return NAN;
    }
    let mut prev_value = rfa.values[0];
    let mut n = 0;
    for v in rfa.values.iter().skip(1) {
        if *v != prev_value {
            n += 1;
            prev_value = *v;
        }
    }

    n as f64
}

pub(super) fn rollup_changes(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut n = 0;
    let mut start = 0;
    if rfa.prev_value.is_nan() {
        if rfa.values.is_empty() {
            return NAN;
        }
        rfa.prev_value = rfa.values[0];
        start = 1;
        n += 1;
    }

    for v in rfa.values.iter().skip(start) {
        if *v != rfa.prev_value {
            n += 1;
            rfa.prev_value = *v;
        }
    }

    return n as f64;
}

pub(super) fn rollup_increases(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
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
    for v in rfa.values.iter().skip(start) {
        if *v > rfa.prev_value {
            n += 1;
        }
        rfa.prev_value = *v;
    }

    return n as f64;
}

// `decreases_over_time` logic is the same as `resets` logic.
const ROLLUP_DECREASES: RollupFunc = rollup_resets;

pub(super) fn rollup_resets(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if rfa.values.is_empty() {
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
    for v in values.iter().skip(start) {
        if *v < prev_value {
            n += 1;
        }
        prev_value = *v;
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
        if i > 0 && rfa.timestamps[i] >= *curr_timestamp {
            i -= 1
        } else {
            if i == 0 {
                return &[];
            }
            break;
        }
    }

    return &rfa.values[0..i];
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
    if values.is_empty() {
        return NAN;
    }
    return values[0];
}

fn rollup_close(rfa: &mut RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return get_first_value_for_candlestick(rfa);
    }
    values[values.len()]
}

fn rollup_high(rfa: &mut RollupFuncArg) -> f64 {
    let mut max = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if max.is_nan() {
        if values.is_empty() {
            return NAN;
        }
        max = values[0];
        start = 1;
    }

    for v in &values[start..] {
        if *v > max {
            max = *v
        }
    }

    max
}

fn rollup_low(rfa: &mut RollupFuncArg) -> f64 {
    let mut min = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if min.is_nan() {
        if values.is_empty() {
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

pub(super) fn rollup_mode_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.

    // Copy rfa.values to a, since modeNoNaNs modifies a contents.
    if rfa.values.is_empty() {
        let mut a = vec![];
        return mode_no_nans(rfa.prev_value, &mut a);
    }
    let mut a = get_float64s(rfa.values.len());
    a.extend(&rfa.values);
    mode_no_nans(rfa.prev_value, &mut a)
}

pub(super) fn rollup_ascent_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    let mut prev_value = rfa.prev_value;
    let mut start: usize = 0;
    if prev_value.is_nan() {
        if values.is_empty() {
            return NAN;
        }
        prev_value = values[0];
        start = 1;
    }
    let mut s: f64 = 0.0;
    for v in &values[start..] {
        let d = v - prev_value;
        if d > 0.0 {
            s += d;
        }
        prev_value = *v;
    }
    return s;
}

pub(super) fn rollup_descent_over_time(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut ofs = 0;
    if rfa.prev_value.is_nan() {
        if rfa.values.is_empty() {
            return NAN;
        }
        rfa.prev_value = rfa.values[0];
        ofs = 1;
    }

    let mut s: f64 = 0.0;
    for v in &rfa.values[ofs..] {
        let d = rfa.prev_value - *v;
        if d > 0.0 {
            s += d;
        }
        rfa.prev_value = *v;
    }

    s
}

pub(super) fn rollup_zscore_over_time(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_first(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let values = &rfa.values;
    if values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    return values[0];
}

pub(crate) fn rollup_default(rfa: &mut RollupFuncArg) -> f64 {
    let values = &rfa.values;
    if values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    // Intentionally do not skip the possible last Prometheus staleness mark.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1526 .
    *values.last().unwrap()
}

pub(super) fn rollup_last(rfa: &mut RollupFuncArg) -> f64 {
    if rfa.values.is_empty() {
        // do not take into account rfa.prev_value, since it may lead
        // to inconsistent results comparing to Prometheus on broken time series
        // with irregular data points.
        return NAN;
    }
    *rfa.values.last().unwrap()
}

pub(super) fn rollup_distinct(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return 0.0;
    }

    rfa.values.sort_by(|a, b| a.total_cmp(&b));
    rfa.values.dedup();

    return rfa.values.len() as f64;
}

pub(super) fn rollup_integrate(rfa: &mut RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    let mut timestamps = &rfa.timestamps[0..];
    let mut prev_value = &rfa.prev_value;
    let mut prev_timestamp = &rfa.curr_timestamp - &rfa.window;
    if prev_value.is_nan() {
        if values.is_empty() {
            return NAN;
        }
        prev_value = &values[0];
        prev_timestamp = timestamps[0];
        values = &values[1..];
        timestamps = &timestamps[1..];
    }

    let mut sum: f64 = 0.0;
    for (v, ts) in values.iter().zip(timestamps.iter()) {
        let dt = (ts - prev_timestamp) as f64 / 1e3_f64;
        sum += prev_value * dt;
        prev_timestamp = *ts;
        prev_value = v;
    }

    let dt = (&rfa.curr_timestamp - prev_timestamp) as f64 / 1e3_f64;
    sum += prev_value * dt;
    return sum;
}

fn rollup_fake(_rfa: &mut RollupFuncArg) -> f64 {
    panic!("BUG: rollup_fake shouldn't be called");
}
