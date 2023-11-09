use std::borrow::Cow;

use rayon::iter::IntoParallelRefIterator;

use metricsql_common::pool::{get_pooled_vec_f64, get_pooled_vec_i64};
use metricsql_parser::ast::{BinaryExpr, DurationExpr};
use metricsql_parser::functions::RollupFunction;

use crate::common::math::is_stale_nan;
use crate::execution::EvalConfig;
use crate::rayon::iter::ParallelIterator;
use crate::{QueryValue, RuntimeResult, Timeseries};

pub(crate) fn series_len(val: &QueryValue) -> usize {
    match &val {
        QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => iv.len(),
        _ => 1,
    }
}

#[inline]
pub fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| !ts.values.iter().all(|v| v.is_nan()));
}

pub fn should_keep_metric_names(be: &BinaryExpr) -> bool {
    if be.op.is_comparison() && !be.returns_bool() {
        // Do not reset MetricGroup for non-boolean `compare` binary ops like Prometheus does.
        return true;
    }
    // Do not reset MetricGroup if it is explicitly requested via `a op b keep_metric_names`
    // See https://docs.victoriametrics.com/MetricsQL.html#keep_metric_names
    be.keep_metric_names()
}

#[inline]
pub(crate) fn adjust_eval_range<'a>(
    func: &RollupFunction,
    offset: &Option<DurationExpr>,
    ec: &'a EvalConfig,
) -> RuntimeResult<(i64, Cow<'a, EvalConfig>)> {
    let mut ec_new = Cow::Borrowed(ec);
    let mut offset: i64 = duration_value(offset, ec.step);
    if offset != 0 {
        let mut result = ec.copy_no_timestamps();
        result.start -= offset;
        result.end -= offset;
        ec_new = Cow::Owned(result);
        // There is no need in calling adjust_start_end() on ec_new if ecNew.may_cache is set to true,
        // since the time range alignment has been already performed by the caller,
        // so cache hit rate should be quite good.
        // See also https://github.com/VictoriaMetrics/VictoriaMetrics/issues/976
    }

    if *func == RollupFunction::RollupCandlestick {
        // Automatically apply `offset -step` to `rollup_candlestick` function
        // in order to obtain expected OHLC results.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309#issuecomment-582113462
        let step = ec_new.step;
        let mut result = ec_new.copy_no_timestamps();
        result.start += step;
        result.end += step;
        offset -= step;
        ec_new = Cow::Owned(result);
    }

    Ok((offset, ec_new))
}

pub(crate) fn duration_value(dur: &Option<DurationExpr>, step: i64) -> i64 {
    if let Some(ofs) = dur {
        ofs.value(step)
    } else {
        0
    }
}

pub(crate) fn get_step(expr: &Option<DurationExpr>, step: i64) -> i64 {
    let res = duration_value(expr, step);
    if res == 0 {
        step
    } else {
        res
    }
}

/// Executes `f` for each `Timeseries` in `tss` in parallel.
pub(super) fn process_series_in_parallel<F>(
    tss: &Vec<Timeseries>,
    f: F,
) -> RuntimeResult<(Vec<Timeseries>, u64)>
where
    F: Fn(&Timeseries, &mut [f64], &[i64]) -> RuntimeResult<(Vec<Timeseries>, u64)> + Send + Sync,
{
    let handler = |ts: &Timeseries| -> RuntimeResult<(Vec<Timeseries>, u64)> {
        let len = ts.values.len();
        // todo: should we have an upper limit here to avoid OOM? Or explicitly size down
        // afterward if needed?
        let mut values = get_pooled_vec_f64(len);
        let mut timestamps = get_pooled_vec_i64(len);

        // todo(perf): have param for if values have NaNs
        remove_nan_values(&mut values, &mut timestamps, &ts.values, &ts.timestamps);

        f(ts, &mut values, &mut timestamps)
    };

    let res = if tss.len() > 1 {
        let res: RuntimeResult<Vec<(Vec<Timeseries>, u64)>> = tss.par_iter().map(handler).collect();
        res?
    } else {
        vec![handler(&tss[0])?]
    };

    let mut series: Vec<Timeseries> = Vec::with_capacity(tss.len());
    let mut sample_total = 0_u64;
    for (timeseries, sample_count) in res.into_iter() {
        sample_total += sample_count;
        series.extend::<Vec<Timeseries>>(timeseries)
    }

    Ok((series, sample_total))
}

pub(crate) fn remove_nan_values(
    dst_values: &mut Vec<f64>,
    dst_timestamps: &mut Vec<i64>,
    values: &[f64],
    timestamps: &[i64],
) {
    let mut has_nan = false;
    for v in values {
        if v.is_nan() {
            has_nan = true;
            break;
        }
    }

    if !has_nan {
        // Fast path - no NaNs.
        dst_values.extend_from_slice(values);
        dst_timestamps.extend_from_slice(timestamps);
        return;
    }

    // Slow path - remove NaNs.
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        dst_values.push(*v);
        dst_timestamps.push(timestamps[i])
    }
}

pub(crate) fn drop_stale_nans(
    func: RollupFunction,
    values: &mut Vec<f64>,
    timestamps: &mut Vec<i64>,
) {
    if func == RollupFunction::DefaultRollup || func == RollupFunction::StaleSamplesOverTime {
        // do not drop Prometheus staleness marks (aka stale NaNs) for default_rollup() function,
        // since it uses them for Prometheus-style staleness detection.
        // do not drop staleness marks for stale_samples_over_time() function, since it needs
        // to calculate the number of staleness markers.
        return;
    }
    // Remove Prometheus staleness marks, so non-default rollup functions don't hit NaN values.
    let has_stale_samples = values.iter().any(|x| is_stale_nan(*x));

    if !has_stale_samples {
        // Fast path: values have no Prometheus staleness marks.
        return;
    }

    // Slow path: drop Prometheus staleness marks from values.
    let mut k = 0;
    for i in 0..values.len() {
        let v = values[i];
        if !is_stale_nan(v) {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
    }

    values.truncate(k);
    timestamps.truncate(k);
}
