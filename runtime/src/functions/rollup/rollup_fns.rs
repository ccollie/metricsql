use std::str::FromStr;

use lib::{get_pooled_vec_f64, is_stale_nan};
use metricsql::functions::RollupFunction;

use crate::common::math::{linear_regression, mad, mode_no_nans, quantile, stddev, stdvar};
use crate::functions::arg_parse::get_scalar_arg_as_vec;
use crate::functions::rollup::quantiles::{new_rollup_quantile, new_rollup_quantiles};
use crate::functions::rollup::types::RollupHandlerFactory;
use crate::functions::rollup::{
    filtered_counts::{
        new_rollup_count_eq, new_rollup_count_gt, new_rollup_count_le, new_rollup_count_ne,
        new_rollup_share_gt, new_rollup_share_le,
    },
    holt_winters::new_rollup_holt_winters,
};
use crate::functions::rollup::{RollupFunc, RollupFuncArg, RollupHandlerEnum};
use crate::functions::types::get_scalar_param_value;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::Timestamp;
use crate::EvalConfig;
use crate::QueryValue;

// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/master/app/vmselect/promql/rollup.go

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

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

pub(crate) fn get_rollup_func_by_name(name: &str) -> RuntimeResult<RollupFunction> {
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

macro_rules! make_factory {
    ( $name: ident, $rf: expr ) => {
        #[inline]
        fn $name(_: &Vec<QueryValue>, _ec: &EvalConfig) -> RuntimeResult<RollupHandlerEnum> {
            Ok(RollupHandlerEnum::wrap($rf))
        }
    };
}

macro_rules! fake_wrapper {
    ( $funcName: ident, $name: expr ) => {
        #[inline]
        fn $funcName(_: &Vec<QueryValue>, _ec: &EvalConfig) -> RuntimeResult<RollupHandlerEnum> {
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

pub(crate) fn rollup_func_keeps_metric_name(name: &str) -> bool {
    match RollupFunction::from_str(name) {
        Err(_) => false,
        Ok(func) => func.keep_metric_name(),
    }
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

fn new_rollup_predict_linear(
    args: &Vec<QueryValue>,
    _ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
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

fn new_rollup_duration_over_time(
    args: &Vec<QueryValue>,
    _ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
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

fn new_rollup_hoeffding_bound_lower(
    args: &Vec<QueryValue>,
    _ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    let phi = get_scalar_param_value(args, 0, "hoeffding_bound_lower", "phi")?;

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let (bound, avg) = hoeffding_bound_internal(&rfa.values, phi);
        avg - bound
    });

    Ok(RollupHandlerEnum::General(f))
}

fn new_rollup_hoeffding_bound_upper(
    args: &Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    let phis = get_scalar_arg_as_vec(args, 0, ec)?;

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        let phi = phis[rfa.idx % phis.len()];
        let (bound, avg) = hoeffding_bound_internal(&rfa.values, phi);
        return avg + bound;
    });

    Ok(RollupHandlerEnum::General(f))
}

fn hoeffding_bound_internal(values: &[f64], phi: f64) -> (f64, f64) {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if values.is_empty() {
        return (NAN, NAN);
    }
    if values.len() == 1 {
        return (0.0, values[0]);
    }

    let (v_avg, v_range) = {
        let mut v_min = values[0];
        let mut v_max = v_min;
        let mut v_sum = 0.0;
        for v in values.iter() {
            if *v < v_min {
                v_min = *v;
            }
            if *v > v_max {
                v_max = *v;
            }
            v_sum += *v;
        }
        let v_avg = v_sum / values.len() as f64;
        let v_range = v_max - v_min;
        (v_avg, v_range)
    };

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
    let bound = v_range * ((1.0 / (1.0 - phi)).ln() / (2 * values.len()) as f64).sqrt();
    return (bound, v_avg);
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
    if rfa.values.is_empty() {
        return NAN;
    }
    let values = &rfa.values;
    let mut max = values[0];
    let mut min = max;
    for v in values.iter() {
        if *v > max {
            max = *v;
        }
        if *v < min {
            min = *v;
        }
    }
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
        return 0f64;
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
    let count = timestamps.len();
    if rfa.prev_value.is_nan() {
        if count < 2 {
            return NAN;
        }
        return (timestamps[count - 1] - timestamps[0]) as f64 / 1e3_f64;
    }
    if count == 0 {
        return NAN;
    }
    return (timestamps[count - 1] - rfa.prev_timestamp) as f64 / 1e3_f64;
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
    if rfa.values.is_empty() {
        if rfa.prev_value.is_nan() {
            return NAN;
        }
        return 0.0;
    }
    let mut prev_value = rfa.prev_value;
    let mut start = 0;
    if prev_value.is_nan() {
        prev_value = rfa.values[0];
        start = 1;
    }

    let values = &rfa.values[start..];
    if values.is_empty() {
        return 0.0;
    }

    let mut n = 0;
    for v in values.iter() {
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

pub(super) fn rollup_open(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_close(rfa: &mut RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return get_first_value_for_candlestick(rfa);
    }
    values[values.len()]
}

pub(super) fn rollup_high(rfa: &mut RollupFuncArg) -> f64 {
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

pub(super) fn rollup_low(rfa: &mut RollupFuncArg) -> f64 {
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
    let mut a = get_pooled_vec_f64(rfa.values.len());
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

pub(super) fn rollup_fake(_rfa: &mut RollupFuncArg) -> f64 {
    panic!("BUG: rollup_fake shouldn't be called");
}
