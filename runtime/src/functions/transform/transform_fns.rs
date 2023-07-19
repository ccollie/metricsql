use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::ops::Deref;

use chrono::Utc;
use num_traits::FloatConst;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal};
use regex::Regex;

use lib::{
    copysign, datetime_part, fmod, from_float, get_float64s, isinf, modf,
    timestamp_secs_to_utc_datetime, DateTimePart,
};
use metricsql::ast::{Expr, FunctionExpr};
use metricsql::functions::TransformFunction;
use metricsql::parser::{compile_regexp, parse_number};

use crate::chrono_tz::Tz;
use crate::eval::{eval_number, eval_time, merge_non_overlapping_timeseries, EvalConfig};
use crate::functions::rollup::{linear_regression, mad, stddev, stdvar};
use crate::functions::utils::{
    float_to_int_bounded, get_first_non_nan_index, get_last_non_nan_index,
};
use crate::functions::{quantile, quantile_sorted};
use crate::rand_distr::Distribution;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{remove_empty_series, MetricName, QueryValue, Timeseries, METRIC_NAME_LABEL};

use super::utils::{get_timezone_offset, ru};

const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

pub struct TransformFuncArg<'a> {
    pub ec: &'a EvalConfig,
    pub fe: &'a FunctionExpr,
    pub args: Vec<QueryValue>,
    pub keep_metric_names: bool,
}

// https://stackoverflow.com/questions/57937436/how-to-alias-an-impl-trait
// https://www.worthe-it.co.za/blog/2017-01-15-aliasing-traits-in-rust.html

// This trait is local to this crate,
// so we can implement it on any type we want.
pub trait TransformFn:
    Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> + Sync + Send
{
}

/// implement `Transform` on any type that implements `Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>>>`.
impl<T> TransformFn for T where
    T: Fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> + Sync + Send
{
}

pub type TransformFuncHandler = fn(&mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>>;

trait TransformValuesFn: FnMut(&mut [f64]) -> () {}
impl<T> TransformValuesFn for T where T: FnMut(&mut [f64]) -> () {}

macro_rules! create_func_one_arg {
    ($name: ident, $func: expr) => {
        pub(crate) fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            fn tfe(values: &mut [f64]) {
                for value in values.iter_mut() {
                    *value = $func(*value)
                }
            }

            transform_series(tfa, tfe)
        }
    };
}

create_func_one_arg!(transform_abs, |x: f64| x.abs());
create_func_one_arg!(transform_acos, |x: f64| x.acos());
create_func_one_arg!(transform_acosh, |x: f64| x.acosh());
create_func_one_arg!(transform_asin, |x: f64| x.asin());
create_func_one_arg!(transform_asinh, |x: f64| x.asinh());
create_func_one_arg!(transform_atan, |x: f64| x.atan());
create_func_one_arg!(transform_atanh, |x: f64| x.atanh());
create_func_one_arg!(transform_ceil, |x: f64| x.ceil());
create_func_one_arg!(transform_cos, |x: f64| x.cos());
create_func_one_arg!(transform_cosh, |x: f64| x.cosh());
create_func_one_arg!(transform_deg, |x: f64| x.to_degrees());
create_func_one_arg!(transform_exp, |x: f64| x.exp());
create_func_one_arg!(transform_floor, |x: f64| x.floor());
create_func_one_arg!(transform_ln, |x: f64| x.ln());
create_func_one_arg!(transform_log2, |x: f64| x.log2());
create_func_one_arg!(transform_log10, |x: f64| x.log10());
create_func_one_arg!(transform_rad, |x: f64| x.to_radians());
create_func_one_arg!(transform_sin, |x: f64| x.sin());
create_func_one_arg!(transform_sinh, |x: f64| x.sinh());
create_func_one_arg!(transform_sqrt, |x: f64| x.sqrt());
create_func_one_arg!(transform_tan, |x: f64| x.tan());
create_func_one_arg!(transform_tanh, |x: f64| x.tanh());

fn transform_sort(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, false)
}

fn transform_sort_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, true)
}

fn transform_sort_alpha_numeric(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, false)
}

fn transform_sort_alpha_numeric_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, true)
}

fn transform_running_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, running_avg)
}

fn transform_running_sum(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, running_sum)
}

fn transform_running_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, running_min)
}

fn transform_running_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    running_func_impl(tfa, running_max)
}

pub fn get_transform_func(f: TransformFunction) -> TransformFuncHandler {
    use TransformFunction::*;
    match f {
        Abs => transform_abs,
        Absent => transform_absent,
        Acos => transform_acos,
        Acosh => transform_acosh,
        Alias => transform_alias,
        Asin => transform_asin,
        Asinh => transform_asinh,
        Atan => transform_atan,
        Atanh => transform_atanh,
        BitmapAnd => transform_bitmap_and,
        BitmapOr => transform_bitmap_or,
        BitmapXor => transform_bitmap_xor,
        BucketsLimit => transform_buckets_limit,
        Ceil => transform_ceil,
        Clamp => transform_clamp,
        ClampMax => transform_clamp_max,
        ClampMin => transform_clamp_min,
        Cos => transform_cos,
        Cosh => transform_cosh,
        DayOfMonth => transform_day_of_month,
        DayOfWeek => transform_day_of_week,
        DaysInMonth => transform_days_in_month,
        Deg => transform_deg,
        DropCommonLabels => transform_drop_common_labels,
        End => transform_end,
        Exp => transform_exp,
        Floor => transform_floor,
        HistogramAvg => transform_histogram_avg,
        HistogramQuantile => transform_histogram_quantile,
        HistogramQuantiles => transform_histogram_quantiles,
        HistogramShare => transform_histogram_share,
        HistogramStddev => transform_histogram_stddev,
        HistogramStdvar => transform_histogram_stdvar,
        Hour => transform_hour,
        Interpolate => transform_interpolate,
        KeepLastValue => transform_keep_last_value,
        KeepNextValue => transform_keep_next_value,
        LabelCopy => transform_label_copy,
        LabelDel => transform_label_del,
        LabelGraphiteGroup => transform_label_graphite_group,
        LabelJoin => transform_label_join,
        LabelKeep => transform_label_keep,
        LabelLowercase => transform_label_lowercase,
        LabelMap => transform_label_map,
        LabelMatch => transform_label_match,
        LabelMismatch => transform_label_mismatch,
        LabelMove => transform_label_move,
        LabelReplace => transform_label_replace,
        LabelSet => transform_label_set,
        LabelTransform => transform_label_transform,
        LabelUppercase => transform_label_uppercase,
        LabelValue => transform_label_value,
        LimitOffset => transform_limit_offset,
        Ln => transform_ln,
        Log2 => transform_log2,
        Log10 => transform_log10,
        Minute => transform_minute,
        Month => transform_month,
        Now => transform_now,
        Pi => transform_pi,
        PrometheusBuckets => transform_prometheus_buckets,
        Rad => transform_rad,
        Random => transform_rand,
        RandExponential => transform_rand_exp,
        RandNormal => transform_rand_norm,
        RangeAvg => transform_range_avg,
        RangeFirst => transform_range_first,
        RangeLast => transform_range_last,
        RangeLinearRegression => transform_range_linear_regression,
        RangeMax => transform_range_max,
        RangeMedian => transform_range_median,
        RangeMin => transform_range_min,
        RangeNormalize => transform_range_normalize,
        RangeQuantile => transform_range_quantile,
        RangeStdDev => transform_range_stddev,
        RangeStdVar => transform_range_stdvar,
        RangeSum => transform_range_sum,
        RangeTrimOutliers => transform_range_trim_outliers,
        RangeTrimSpikes => transform_range_trim_spikes,
        RangeTrimZscore => transform_range_trim_zscore,
        RangeZscore => transform_range_zscore,
        RemoveResets => transform_remove_resets,
        Round => transform_round,
        Ru => transform_range_ru,
        RunningAvg => transform_running_avg,
        RunningMax => transform_running_max,
        RunningMin => transform_running_min,
        RunningSum => transform_running_sum,
        Scalar => transform_scalar,
        Sgn => transform_sgn,
        Sin => transform_sin,
        Sinh => transform_sinh,
        SmoothExponential => transform_smooth_exponential,
        Sort => transform_sort,
        SortByLabel => transform_sort_by_label,
        SortByLabelDesc => transform_sort_by_label_desc,
        SortByLabelNumeric => transform_sort_alpha_numeric,
        SortByLabelNumericDesc => transform_sort_alpha_numeric_desc,
        SortDesc => transform_sort_desc,
        Sqrt => transform_sqrt,
        Start => transform_start,
        Step => transform_step,
        Tan => transform_tan,
        Tanh => transform_tanh,
        Time => transform_time,
        TimezoneOffset => transform_timezone_offset,
        Union => transform_union,
        Vector => transform_vector,
        Year => transform_year,
    }
}

fn transform_series(
    tfa: &mut TransformFuncArg,
    tf: impl TransformValuesFn,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    do_transform_values(&mut series, tf, tfa.keep_metric_names)
}

fn do_transform_values(
    arg: &mut Vec<Timeseries>,
    mut tf: impl TransformValuesFn,
    keep_metric_names: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    for ts in arg.iter_mut() {
        if !keep_metric_names {
            ts.metric_name.reset_metric_group();
        }
        tf(&mut ts.values);
    }
    Ok(std::mem::take(arg))
}

fn transform_absent(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = get_absent_timeseries(&mut tfa.ec, &tfa.fe.args[0])?;

    let series = get_series(tfa, 0)?;
    if series.len() == 0 {
        return Ok(rvs);
    }

    for i in 0..series[0].values.len() {
        let mut is_absent = true;
        for ts in series.iter() {
            if !ts.values[i].is_nan() {
                is_absent = false;
                break;
            }
        }
        if !is_absent {
            rvs[0].values[i] = NAN
        }
    }
    return Ok(rvs);
}

pub(crate) fn get_absent_timeseries(ec: &EvalConfig, arg: &Expr) -> RuntimeResult<Vec<Timeseries>> {
    // Copy tags from arg
    let mut rvs = eval_number(ec, 1.0)?;
    match arg {
        Expr::MetricExpression(me) => {
            for tf in me.label_filters.iter() {
                if tf.label.len() == 0 {
                    continue;
                }
                if tf.is_regexp() || tf.is_negative() {
                    continue;
                }
                rvs[0].metric_name.set_tag(&tf.label, &tf.value)
            }
        }
        _ => {}
    }
    Ok(rvs)
}

fn transform_clamp(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mins = get_scalar(tfa, 1)?;
    let maxs = get_scalar(tfa, 2)?;
    // todo: are these guaranteed to be of equal length ?
    let tf = |values: &mut [f64]| {
        for ((v, min), max) in values.iter_mut().zip(mins.iter()).zip(maxs.iter()) {
            if *v < *min {
                *v = *min;
            } else if *v > *max {
                *v = *max;
            }
        }
    };

    transform_series(tfa, tf)
}

fn transform_clamp_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let maxs = get_scalar(tfa, 1)?;
    let tf = |values: &mut [f64]| {
        for (v, max) in values.iter_mut().zip(maxs.iter()) {
            if *v > *max {
                *v = *max;
            }
        }
    };

    transform_series(tfa, tf)
}

fn transform_clamp_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mins = get_scalar(tfa, 1)?;
    let tf = |values: &mut [f64]| {
        for (v, min) in values.iter_mut().zip(mins.iter()) {
            if *v < *min {
                *v = *min;
            }
        }
    };

    transform_series(tfa, tf)
}

fn extract_datetime_part(epoch_secs: f64, part: DateTimePart) -> f64 {
    if !epoch_secs.is_nan() {
        if let Some(utc) = timestamp_secs_to_utc_datetime(epoch_secs as i64) {
            if let Some(value) = datetime_part(utc, part) {
                return value as f64;
            }
        }
    }
    f64::NAN
}

fn transform_datetime_impl(
    tfa: &mut TransformFuncArg,
    part: DateTimePart,
) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        for v in values.iter_mut() {
            *v = extract_datetime_part(*v, part)
        }
    };

    let mut arg = if tfa.args.len() == 0 {
        eval_time(&tfa.ec)?
    } else {
        get_series(tfa, 0)?
    };

    do_transform_values(&mut arg, tf, tfa.keep_metric_names)
}

fn transform_hour(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Hour)
}

fn transform_day_of_month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DayOfMonth)
}

fn transform_day_of_week(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DayOfWeek)
}

fn transform_days_in_month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::DaysInMonth)
}

fn transform_minute(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Minute)
}

fn transform_month(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Month)
}

fn transform_year(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_datetime_impl(tfa, DateTimePart::Year)
}

fn transform_buckets_limit(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut limit = get_int_number(&tfa, 1)?;

    if limit <= 0 {
        return Ok(vec![]);
    }
    if limit < 3 {
        // Preserve the first and the last bucket for better accuracy for min and max values.
        limit = 3
    }
    let series = get_series(tfa, 1)?;
    let mut tss = vmrange_buckets_to_le(series);
    let tss_len = tss.len();

    if tss_len == 0 {
        return Ok(vec![]);
    }

    let points_count = tss[0].values.len();

    // Group timeseries by all MetricGroup+tags excluding `le` tag.
    struct Bucket {
        le: f64,
        hits: f64,
        ts_index: usize,
    }

    let mut bucket_map: HashMap<String, Vec<Bucket>> = HashMap::new();

    let mut mn: MetricName = MetricName::default();

    for (ts_index, ts) in tss.iter().enumerate() {
        let le_str = ts.metric_name.get_tag_value("le");

        // Skip time series without `le` tag.
        match le_str {
            None => continue,
            Some(le_str) => {
                if le_str.len() == 0 {
                    continue;
                }
            }
        }

        let le_str = le_str.unwrap();

        if let Ok(le) = le_str.parse::<f64>() {
            mn.copy_from(&ts.metric_name);
            mn.remove_tag("le");

            let key = ts.metric_name.to_string();

            bucket_map.entry(key).or_default().push(Bucket {
                le,
                hits: 0.0,
                ts_index,
            });
        } else {
            // Skip time series with invalid `le` tag.
            continue;
        }
    }

    // Remove buckets with the smallest counters.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss_len);
    for (_, le_group) in bucket_map.iter_mut() {
        if le_group.len() <= limit as usize {
            // Fast path - the number of buckets doesn't exceed the given limit.
            // Keep all the buckets as is.
            let series = le_group
                .into_iter()
                .map(|x| tss.remove(x.ts_index))
                .collect::<Vec<_>>();
            rvs.extend(series);
            continue;
        }
        // Slow path - remove buckets with the smallest number of hits until their count reaches the limit.

        // Calculate per-bucket hits.
        le_group.sort_by(|a, b| a.le.total_cmp(&b.le));
        for n in 0..points_count {
            let mut prev_value: f64 = 0.0;
            for bucket in le_group.iter_mut() {
                if let Some(ts) = tss.get(bucket.ts_index) {
                    let value = ts.values[n];
                    bucket.hits += value - prev_value;
                    prev_value = value
                }
            }
        }
        while le_group.len() > limit as usize {
            // Preserve the first and the last bucket for better accuracy for min and max values
            let mut xx_min_idx = 1;
            let mut min_merge_hits = le_group[1].hits + le_group[2].hits;
            for i in 0..le_group[1..le_group.len() - 2].len() {
                let merge_hits = le_group[i + 1].hits + le_group[i + 2].hits;
                if merge_hits < min_merge_hits {
                    xx_min_idx = i + 1;
                    min_merge_hits = merge_hits
                }
            }
            le_group[xx_min_idx + 1].hits += le_group[xx_min_idx].hits;
            // remove item at xx_min_idx ?
            // leGroup = append(leGroup[: xx_min_idx], leGroup[xx_min_idx + 1: ]...)
            le_group.remove(xx_min_idx);
        }

        let ts_iter = le_group
            .into_iter()
            .map(|x| tss.remove(x.ts_index))
            .collect::<Vec<Timeseries>>();

        rvs.extend(ts_iter);
    }

    Ok(rvs)
}

fn transform_prometheus_buckets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series(tfa, 0)?;
    let rvs = vmrange_buckets_to_le(series);
    return Ok(rvs);
}

static ELLIPSIS: &str = "...";

/// Group timeseries by MetricGroup+tags excluding `vmrange` tag.
struct Bucket {
    start_str: String,
    end_str: String,
    start: f64,
    end: f64,
    ts: Timeseries,
}

impl Bucket {
    fn new(ts: Timeseries) -> Self {
        Self {
            start_str: "".to_string(),
            end_str: "".to_string(),
            start: 0.0,
            end: 0.0,
            ts,
        }
    }
}

impl Default for Bucket {
    fn default() -> Self {
        Self::new(Timeseries::default())
    }
}

pub(crate) fn vmrange_buckets_to_le(tss: Vec<Timeseries>) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());

    let mut buckets: HashMap<String, Vec<Bucket>> = HashMap::new();

    for ts in tss.into_iter() {
        let vm_range = match ts.metric_name.get_tag_value("vmrange") {
            Some(value) => value,
            None => "",
        };

        if vm_range.len() == 0 {
            if let Some(le) = ts.metric_name.get_tag_value("le") {
                if le.len() > 0 {
                    // Keep Prometheus-compatible buckets.
                    rvs.push(ts);
                }
            }
            continue;
        }

        let n = match vm_range.find(ELLIPSIS) {
            Some(pos) => pos,
            None => continue,
        };

        let start_str = &vm_range[0..n];
        let start = match start_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        let end_str = &vm_range[(n + ELLIPSIS.len())..vm_range.len()];
        let end = match end_str.parse::<f64>() {
            Err(_) => continue,
            Ok(n) => n,
        };

        let mut _ts = ts;
        _ts.metric_name.remove_tag("le");
        _ts.metric_name.remove_tag("vmrange");

        let key = _ts.metric_name.to_string();
        // series.push(_ts);

        buckets.entry(key).or_default().push(Bucket {
            start_str: format!("{}", start),
            end_str: format!("{}", end),
            start,
            end,
            ts: std::mem::take(&mut _ts), // does this copy ???
        });
    }

    // Convert `vmrange` label in each group of time series to `le` label.
    let copy_ts = |src: &Timeseries, le_str: &str| -> Timeseries {
        let mut ts: Timeseries = src.clone();
        ts.values.resize(ts.values.len(), 0.0);
        ts.metric_name.remove_tag("le");
        ts.metric_name.set_tag("le", le_str);
        return ts;
    };

    let is_zero_ts = |ts: &Timeseries| -> bool { ts.values.iter().all(|x| *x <= 0.0) };

    let default_bucket: Bucket = Default::default();

    for xss in buckets.values_mut() {
        xss.sort_by(|a, b| a.end.total_cmp(&b.end));
        let mut xss_new: Vec<Bucket> = Vec::with_capacity(xss.len() + 2);
        let mut xs_prev: &Bucket = &default_bucket;
        let mut has_non_empty = false;

        let mut uniq_ts: HashMap<String, usize> = HashMap::with_capacity(xss.len());
        for mut xs in xss.into_iter() {
            if is_zero_ts(&xs.ts) {
                // Skip time series with zeros. They are substituted by xss_new below.
                // Skip buckets with zero values - they will be merged into a single bucket
                // when the next non-zero bucket appears.

                // Do not store xs in xsPrev in order to properly create `le` time series
                // for zero buckets.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/pull/4021
                continue;
            }

            if xs.start != xs_prev.end {
                // There is a gap between the previous bucket and the current bucket
                // or the previous bucket is skipped because it was zero.
                // Fill it with a time series with le=xs.start.
                xs_prev = xs;
                if !uniq_ts.contains_key(&xs.end_str) {
                    let copy = copy_ts(&xs.ts, &xs.end_str);
                    uniq_ts.insert(xs.end_str.to_string(), xss_new.len());
                    xss_new.push(Bucket {
                        start_str: "".to_string(),
                        start: 0.0,
                        end_str: xs.start_str.clone(),
                        end: xs.start,
                        ts: copy,
                    });
                }
                continue;
            }

            // Convert the current time series to a time series with le=xs.end
            xs.ts.metric_name.set_tag("le", &xs.end_str);

            let end_str = xs.end_str.clone();
            match uniq_ts.get(&end_str) {
                Some(prev_index) => {
                    if let Some(prev_bucket) = xss_new.get_mut(*prev_index) {
                        // the end of the current bucket is not unique, need to merge it with the existing bucket.
                        merge_non_overlapping_timeseries(&mut prev_bucket.ts, &xs.ts);
                    }
                }
                None => {
                    xss_new.push(std::mem::take(&mut xs));
                    uniq_ts.insert(xs.end_str.clone(), xss_new.len() - 1);
                }
            }

            has_non_empty = true;
            if xs.start != xs_prev.end && !uniq_ts.contains_key(&xs.start_str) {
                xss_new.push(Bucket {
                    start_str: "".to_string(),
                    end_str: xs.start_str.clone(),
                    start: 0.0,
                    end: xs.start,
                    ts: copy_ts(&xs.ts, &xs.start_str),
                });
                uniq_ts.insert(xs.start_str.clone(), xss_new.len() - 1);
            }
            xs.ts.metric_name.set_tag("le", &xs.end_str);

            xs_prev = xs
        }

        if !has_non_empty {
            xss_new.clear();
            continue;
        }

        if !isinf(xs_prev.end, 1) {
            xss_new.push(Bucket {
                start_str: "".to_string(),
                end_str: "+Inf".to_string(),
                start: 0.0,
                end: f64::INFINITY,
                ts: copy_ts(&xs_prev.ts, "+Inf"),
            })
        }

        if xss_new.len() == 0 {
            continue;
        }

        for i in 0..xss_new[0].ts.values.len() {
            let mut count: f64 = 0.0;
            for xs in xss_new.iter_mut() {
                let v = xs.ts.values[i];
                if !v.is_nan() && v > 0.0 {
                    count += v
                }
                xs.ts.values[i] = count
            }
        }

        for mut xs in xss_new.into_iter() {
            rvs.push(std::mem::take(&mut xs.ts))
        }
    }

    return rvs;
}

fn transform_histogram_share(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let les: Vec<f64> = get_scalar(&tfa, 0)?;

    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let series = get_series(tfa, 1)?;
    let mut tss = vmrange_buckets_to_le(series);

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label = if tfa.args.len() > 2 {
        tfa.args[2].get_string()?
    } else {
        "".to_string()
    };

    // Group metrics by all tags excluding "le"
    let m = group_le_timeseries(&mut tss);

    // Calculate share for les
    let share = |i: usize, les: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let le_req = les[i];
        if le_req.is_nan() || xss.len() == 0 {
            return (NAN, NAN, NAN);
        }
        fix_broken_buckets(i, xss);
        if le_req < 0.0 {
            return (0.0, 0.0, 0.0);
        }
        if isinf(le_req, 1) {
            return (1.0, 1.0, 1.0);
        }
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;

        for xs in xss.iter() {
            let v = xs.ts.values[i];
            let le = xs.le;
            if le_req >= le {
                v_prev = v;
                le_prev = le;
                continue;
            }
            // precondition: le_prev <= le_req < le
            let v_last = xss[xss.len() - 1].ts.values[i];
            let lower = v_prev / v_last;
            if isinf(le, 1) {
                return (lower, lower, 1.0);
            }
            if le_prev == le_req {
                return (lower, lower, lower);
            }
            let upper = v / v_last;
            let q = lower + (v - v_prev) / v_last * (le_req - le_prev) / (le - le_prev);
            return (q, lower, upper);
        }
        // precondition: le_req > leLast
        return (1.0, 1.0, 1.0);
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));

        merge_same_le(&mut xss);

        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if bounds_label.len() > 0 {
            ts_lower = xss[0].ts.clone();
            ts_lower.metric_name.remove_tag(&bounds_label);
            ts_lower.metric_name.set_tag(&bounds_label, "lower");

            ts_upper = xss[0].ts.clone();
            ts_upper.metric_name.remove_tag(&bounds_label);
            ts_upper.metric_name.set_tag(&bounds_label, "upper")
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }

        for i in 0..xss[0].ts.values.len() {
            let (q, lower, upper) = share(i, &les, &mut xss);
            xss[0].ts.values[i] = q;
            if bounds_label.len() > 0 {
                ts_lower.values[i] = lower;
                ts_upper.values[i] = upper
            }
        }

        rvs.push(std::mem::take(&mut xss[0].ts));
        if bounds_label.len() > 0 {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }

    return Ok(rvs);
}

fn transform_histogram_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series(tfa, 0)?;
    let mut tss = vmrange_buckets_to_le(series);
    let mut m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, xss) in m.iter_mut() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            xss[0].ts.values[i] = avg_for_le_timeseries(i, xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn transform_histogram_stddev(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series(tfa, 0)?;
    let mut tss = vmrange_buckets_to_le(series);
    let m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());

    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            let v = stdvar_for_le_timeseries(i, &xss);
            xss[0].ts.values[i] = v.sqrt();
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn transform_histogram_stdvar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series(tfa, 0)?;
    let mut tss = vmrange_buckets_to_le(series);
    let m = group_le_timeseries(&mut tss);
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for (_, mut xss) in m.into_iter() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        for i in 0..xss[0].ts.values.len() {
            xss[0].ts.values[i] = stdvar_for_le_timeseries(i, &xss)
        }
        rvs.push(std::mem::take(&mut xss[0].ts));
    }
    return Ok(rvs);
}

fn avg_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if isinf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = f64::from(le + le_prev) / 2_f64;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v;
    }
    if weight_total == 0.0 {
        return NAN;
    }
    return sum / weight_total;
}

fn stdvar_for_le_timeseries(i: usize, xss: &[LeTimeseries]) -> f64 {
    let mut le_prev: f64 = 0.0;
    let mut v_prev: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut weight_total: f64 = 0.0;
    for xs in xss {
        if isinf(xs.le, 0) {
            continue;
        }
        let le = xs.le;
        let n = (le + le_prev) / 2.0;
        let v = xs.ts.values[i];
        let weight = v - v_prev;
        sum += n * weight;
        sum2 += n * n * weight;
        weight_total += weight;
        le_prev = le;
        v_prev = v
    }
    if weight_total == 0.0 {
        return NAN;
    }
    let avg = sum / weight_total;
    let avg2 = sum2 / weight_total;
    let mut stdvar = avg2 - avg * avg;
    if stdvar < 0.0 {
        // Correct possible calculation error.
        stdvar = 0.0
    }
    return stdvar;
}

fn transform_range_normalize(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs: Vec<Timeseries> = vec![];
    let mut selected: Vec<usize> = Vec::with_capacity(tfa.args.len());
    for i in 0..tfa.args.len() {
        let mut series = get_series(tfa, i)?; // todo: get_matrix
        for (j, ts) in series.iter_mut().enumerate() {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for v in ts.values.iter() {
                if v.is_nan() {
                    continue;
                }
                min = min.min(*v);
                max = max.max(*v);
            }
            let d = max - min;
            if d.is_infinite() {
                continue;
            }
            for v in ts.values.iter_mut() {
                *v = (*v - min) / d;
            }
            selected.push(j);
        }

        selected.iter().for_each(|i| {
            rvs.push(series.remove(*i));
        });

        selected.clear();
    }
    Ok(rvs)
}

fn transform_range_trim_zscore(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let z = get_scalar_float(&tfa, 0, Some(0_f64))?.abs();

    // Trim samples with z-score above z.
    let mut rvs = get_series(tfa, 1)?;
    for ts in rvs.iter_mut() {
        // todo: use rapid calculation methods for mean and stddev.
        let q_stddev = stddev(&ts.values);
        let avg = mean(&ts.values);
        for v in ts.values.iter_mut() {
            let z_curr = (*v - avg).abs() / q_stddev;
            if z_curr > z {
                *v = f64::NAN
            }
        }
    }
    return Ok(rvs);
}

fn transform_range_zscore(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 1)?;
    let mut rvs = get_series(tfa, 1)?;
    for ts in rvs.iter_mut() {
        // todo: use rapid calculation methods for mean and stddev.
        let q_stddev = stddev(&ts.values);
        let avg = mean(&ts.values);
        for v in ts.values.iter_mut() {
            *v = (*v - avg) / q_stddev
        }
    }
    Ok(rvs)
}

fn mean(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut n = 0;
    for v in values.iter() {
        if v.is_nan() {
            continue;
        }
        sum += v;
        n += 1;
    }
    return sum / n as f64;
}

fn transform_range_trim_outliers(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let k = get_scalar_float(tfa, 0, Some(0_f64))?;

    // Trim samples satisfying the `abs(v - range_median(q)) > k*range_mad(q)`
    let mut rvs = get_series(tfa, 1)?;
    for ts in rvs.iter_mut() {
        let d_max = k * mad(&ts.values);
        let q_median = quantile(0.5, &ts.values);
        for v in ts.values.iter_mut() {
            if (*v - q_median).abs() > d_max {
                *v = f64::NAN
            }
        }
    }
    return Ok(rvs);
}

fn transform_range_trim_spikes(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    expect_transform_args_num(tfa, 2)?;
    let mut phi = get_scalar_float(tfa, 0, Some(0_f64))?;

    // Trim 100% * (phi / 2) samples with the lowest / highest values per each time series
    phi /= 2.0;
    let phi_upper = 1.0 - phi;
    let phi_lower = phi;
    let mut rvs = get_series(tfa, 1)?;
    let value_count = rvs[0].values.len();
    let mut values = get_float64s(value_count);

    for ts in rvs.iter_mut() {
        values.clear();
        for v in ts.values.iter() {
            if !v.is_nan() {
                values.push(*v);
            }
        }

        values.sort_by(|a, b| a.total_cmp(&b));

        let v_max = quantile_sorted(phi_upper, &values);
        let v_min = quantile_sorted(phi_lower, &values);
        for v in ts.values.iter_mut() {
            if !v.is_nan() {
                if *v > v_max || *v < v_min {
                    *v = f64::NAN;
                }
            }
        }
    }

    Ok(rvs)
}

fn transform_range_linear_regression(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let timestamps = ts.timestamps.deref();
        if timestamps.len() == 0 {
            continue;
        }
        let intercept_timestamp = timestamps[0];
        let (v, k) = linear_regression(&ts.values, &timestamps, intercept_timestamp);
        for (value, timestamp) in ts.values.iter_mut().zip(timestamps.iter()) {
            *value = v + k * ((timestamp - intercept_timestamp) as f64 / 1e3)
        }
    }

    return Ok(series);
}

fn transform_range_stddev(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let dev = stddev(&ts.values);
        for v in ts.values.iter_mut() {
            *v = dev;
        }
    }
    return Ok(series);
}

fn transform_range_stdvar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?; // todo: get_matrix
    for ts in series.iter_mut() {
        let v = stdvar(&ts.values);
        for v1 in ts.values.iter_mut() {
            *v1 = v
        }
    }
    return Ok(series);
}

fn transform_histogram_quantiles(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = tfa.args[0].get_string()?;

    let len = tfa.args.len();
    let tss_orig = tfa.args[len - 1].as_instant_vec(tfa.ec)?;
    // Calculate quantile individually per each phi.
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tfa.args.len());

    let mut tfa_tmp = TransformFuncArg {
        ec: tfa.ec,
        fe: tfa.fe,
        args: vec![],
        keep_metric_names: tfa.keep_metric_names,
    };

    for i in 1..len - 1 {
        let phi_arg = get_scalar_float(tfa, i, Some(0_f64))?;
        if phi_arg < 0.0 || phi_arg > 1.0 {
            let msg = "got unexpected phi arg. it should contain only numbers in the range [0..1]";
            return Err(RuntimeError::ArgumentError(msg.to_string()));
        }
        let phi_str = phi_arg.to_string();
        let tss = copy_timeseries(&tss_orig);

        tfa_tmp.args = vec![QueryValue::Scalar(phi_arg), QueryValue::InstantVector(tss)];

        match transform_histogram_quantile(&mut tfa_tmp) {
            Err(e) => {
                let msg = format!("cannot calculate quantile {}: {:?}", phi_str, e);
                return Err(RuntimeError::General(msg));
            }
            Ok(mut tss_tmp) => {
                for ts in tss_tmp.iter_mut() {
                    ts.metric_name.remove_tag(&dst_label);
                    ts.metric_name.set_tag(&dst_label, &phi_str);
                }
                rvs.extend(tss_tmp)
            }
        }
    }

    Ok(rvs)
}

fn transform_histogram_quantile(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let phis: Vec<f64> = get_scalar(&tfa, 0)?;

    // Convert buckets with `vmrange` labels to buckets with `le` labels.
    let series = get_series(tfa, 1)?;
    let mut tss = vmrange_buckets_to_le(series);

    // Parse bounds_label. See https://github.com/prometheus/prometheus/issues/5706 for details.
    let bounds_label = if tfa.args.len() > 2 {
        tfa.args[2].get_string()?
    } else {
        "".to_string()
    };

    // Group metrics by all tags excluding "le"
    let m = group_le_timeseries(&mut tss);

    // Calculate quantile for each group in m
    let last_non_inf = |_i: usize, xss: &[LeTimeseries]| -> f64 {
        let mut cur = xss;
        while cur.len() > 0 {
            let xs_last = &cur[cur.len() - 1];
            if !isinf(xs_last.le, 0) {
                return xs_last.le;
            }
            cur = &cur[0..cur.len() - 1]
        }
        return NAN;
    };

    let quantile = |i: usize, phis: &[f64], xss: &mut Vec<LeTimeseries>| -> (f64, f64, f64) {
        let phi = phis[i];
        if phi.is_nan() {
            return (NAN, NAN, NAN);
        }
        fix_broken_buckets(i, xss);
        let mut v_last: f64 = 0.0;
        if xss.len() > 0 {
            v_last = xss[xss.len() - 1].ts.values[i]
        }
        if v_last == 0.0 {
            return (NAN, NAN, NAN);
        }
        if phi < 0.0 {
            return (f64::NEG_INFINITY, f64::NEG_INFINITY, xss[0].ts.values[i]);
        }
        if phi > 1.0 {
            return (INF, v_last, INF);
        }
        let v_req = v_last * phi;
        let mut v_prev: f64 = 0.0;
        let mut le_prev: f64 = 0.0;
        for xs in xss.iter() {
            let v = xs.ts.values[i];
            let le = xs.le;
            if v <= 0.0 {
                // Skip zero buckets.
                le_prev = le;
                continue;
            }
            if v < v_req {
                v_prev = v;
                le_prev = le;
                continue;
            }
            if isinf(le, 0) {
                break;
            }
            if v == v_prev {
                return (le_prev, le_prev, v);
            }
            let vv = le_prev + (le - le_prev) * (v_req - v_prev) / (v - v_prev);
            return (vv, le_prev, le);
        }
        let vv = last_non_inf(i, xss);
        return (vv, vv, INF);
    };

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(m.len());
    for mut xss in m.into_values() {
        xss.sort_by(|a, b| a.le.total_cmp(&b.le));
        merge_same_le(&mut xss);

        let mut ts_lower: Timeseries;
        let mut ts_upper: Timeseries;

        if bounds_label.len() > 0 {
            ts_lower = xss[0].ts.clone();
            ts_lower.metric_name.set_tag(&bounds_label, "lower");

            ts_upper = xss[0].ts.clone();
            ts_upper.metric_name.set_tag(&bounds_label, "upper");
        } else {
            ts_lower = Timeseries::default();
            ts_upper = Timeseries::default();
        }

        for (i, (ts_lower, ts_upper)) in ts_lower
            .values
            .iter_mut()
            .zip(ts_upper.values.iter_mut())
            .enumerate()
        {
            let (v, lower, upper) = quantile(i, &phis, &mut xss);
            xss[0].ts.values[i] = v;
            if bounds_label.len() > 0 {
                *ts_lower = lower;
                *ts_upper = upper;
            }
        }

        let mut dst: LeTimeseries = if xss.len() == 1 {
            xss.remove(0)
        } else {
            xss.swap_remove(0)
        };

        rvs.push(std::mem::take(&mut dst.ts));
        if bounds_label.len() > 0 {
            rvs.push(ts_lower);
            rvs.push(ts_upper);
        }
    }

    Ok(rvs)
}

#[derive(Default)]
pub(super) struct LeTimeseries {
    pub le: f64,
    pub ts: Timeseries,
}

// impl<'a> Default for LeTimeseries<'a> {
//     fn default() -> Self {
//         Self {
//             le: 0.0,
//             ts: &mut Timeseries {
//                 metric_name: Default::default(),
//                 values: vec![],
//                 timestamps: Arc::new(vec![])
//             }
//         }
//     }
// }

fn group_le_timeseries(tss: &mut Vec<Timeseries>) -> HashMap<String, Vec<LeTimeseries>> {
    let mut m: HashMap<String, Vec<LeTimeseries>> = HashMap::new();

    for mut ts in tss.iter_mut() {
        match ts.metric_name.get_tag_value("le") {
            None => {}
            Some(tag_value) => {
                if tag_value.len() == 0 {
                    continue;
                }
                match tag_value.parse::<f64>() {
                    Err(..) => continue,
                    Ok(le) => {
                        ts.metric_name.reset_metric_group();
                        ts.metric_name.remove_tag("le");
                        let key = ts.metric_name.to_string();

                        m.entry(key).or_default().push(LeTimeseries {
                            le,
                            ts: std::mem::take(&mut ts),
                        });
                    }
                }
            }
        }
    }

    m
}

pub(super) fn fix_broken_buckets(i: usize, xss: &mut Vec<LeTimeseries>) {
    // Buckets are already sorted by le, so their values must be in ascending order,
    // since the next bucket includes all the previous buckets.
    // If the next bucket has lower value than the current bucket,
    // then the current bucket must be substituted with the next bucket value.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2819
    if xss.len() < 2 {
        return;
    }
    let mut j = xss.len() - 1;
    loop {
        let v = xss[j].ts.values[i];
        if !v.is_nan() {
            j += 1;
            while j < xss.len() {
                xss[j].ts.values[i] = v;
                j += 1;
            }
            break;
        }
        if i == 0 {
            break;
        }
        j -= 1;
    }

    let mut v_next = xss[xss.len() - 1].ts.values[i];

    let mut j = xss.len() - 1;
    loop {
        let v = xss[j].ts.values[i];
        if v.is_nan() || v > v_next {
            xss[j].ts.values[i] = v_next
        } else {
            v_next = v;
        }
        if j == 0 {
            break;
        }
        j -= 1;
    }
}

fn merge_same_le(xss: &mut Vec<LeTimeseries>) -> Vec<LeTimeseries> {
    // Merge buckets with identical le values.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/pull/3225
    let mut prev_le = xss[0].le;
    let mut dst = Vec::with_capacity(xss.len());
    let mut iter = xss.into_iter();
    let first = iter.next();
    if first.is_none() {
        return dst;
    }
    dst.push(std::mem::take(&mut first.unwrap()));
    let mut dst_index = 0;

    for mut xs in iter {
        if xs.le != prev_le {
            prev_le = xs.le.clone();
            dst.push(std::mem::take(&mut xs));
            dst_index = dst.len() - 1;
            continue;
        }

        if let Some(dst) = dst.get_mut(dst_index) {
            for (v, dst_val) in xs.ts.values.iter().zip(dst.ts.values.iter_mut()) {
                *dst_val += v;
            }
        }
    }
    return dst;
}

fn transform_range_ru(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() != 2 {
        // error
    }
    return match (&tfa.args[0], &tfa.args[1]) {
        (QueryValue::Scalar(left), QueryValue::Scalar(right)) => {
            // slight optimization
            let value = ru(*left, *right);
            eval_number(&tfa.ec, value)
        }
        _ => {
            let mut free_series = get_series(tfa, 0)?; // todo: get_range_vector
            let max_series = get_series(tfa, 1)?; // todo: get_range_vector

            for (free_ts, max_ts) in free_series.iter_mut().zip(max_series) {
                // calculate utilization and store in `free`
                for (free_value, max_value) in free_ts.values.iter_mut().zip(max_ts.values) {
                    *free_value = ru(*free_value, max_value)
                }
            }

            Ok(free_series)
        }
    };
}

#[inline]
fn running_sum(a: f64, b: f64, _idx: usize) -> f64 {
    a + b
}

#[inline]
fn running_max(a: f64, b: f64, _idx: usize) -> f64 {
    a.max(b)
}

#[inline]
fn running_min(a: f64, b: f64, _idx: usize) -> f64 {
    a.min(b)
}

#[inline]
fn running_avg(a: f64, b: f64, idx: usize) -> f64 {
    // See `Rapid calculation methods` at https://en.wikipedia.org/wiki/Standard_deviation
    a + (b - a) / (idx + 1) as f64
}

fn transform_keep_last_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        if ts.is_empty() {
            continue;
        }
        let mut last_value = ts.values[0];
        for v in ts.values.iter_mut() {
            if !v.is_nan() {
                last_value = *v;
                continue;
            }
            *v = last_value
        }
    }

    Ok(series)
}

fn transform_keep_next_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        if ts.is_empty() {
            continue;
        }
        let mut next_value = *ts.values.last().unwrap();
        for v in ts.values.iter_mut().rev() {
            if !v.is_nan() {
                next_value = *v;
                continue;
            }
            *v = next_value;
        }
    }

    Ok(series)
}

fn transform_interpolate(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_series(tfa, 0)?;
    for ts in tss.iter_mut() {
        if ts.len() == 0 {
            continue;
        }

        // skip leading and trailing NaNs
        let mut i = 0;
        for v in ts.values.iter() {
            if v.is_nan() {
                i += 1;
            }
            break;
        }

        let mut j = ts.values.len() - 1;
        for v in ts.values.iter().rev() {
            if v.is_nan() {
                if j > i {
                    j -= 1;
                } else {
                    continue;
                }
            }
            break;
        }

        let values = &mut ts.values[i..j];

        let mut i = 0;
        let mut prev_value = f64::NAN;
        let mut next_value: f64;

        while i < values.len() {
            let v = values[i];
            if !v.is_nan() {
                i += 1;
                continue;
            }
            if i > 0 {
                prev_value = values[i - 1]
            }
            let mut j = i + 1;
            while j < values.len() && values[j].is_nan() {
                j += 1;
            }
            if j >= values.len() {
                next_value = prev_value
            } else {
                next_value = values[j]
            }
            if prev_value.is_nan() {
                prev_value = next_value
            }
            let delta = (next_value - prev_value) / (j - i + 1) as f64;
            while i < j {
                prev_value += delta;
                values[i] = prev_value;
                i += 1;
            }
        }
    }

    Ok(tss)
}

fn running_func_impl(
    tfa: &mut TransformFuncArg,
    rf: fn(a: f64, b: f64, idx: usize) -> f64,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut res = get_series(tfa, 0)?;
    for ts in res.iter_mut() {
        ts.metric_name.reset_metric_group();

        // skip NaN values
        let mut start = 0;
        for (i, v) in ts.values.iter_mut().enumerate() {
            if !v.is_nan() {
                start = i;
                break;
            }
        }

        // make sure there's at least 2 items remaining
        if ts.values.len() - start < 2 {
            continue;
        }

        let mut prev_value = ts.values[start];
        for (i, v) in ts.values[start + 1..].iter_mut().enumerate() {
            if !v.is_nan() {
                prev_value = rf(prev_value, *v, i + 1);
            }
            *v = prev_value;
        }
    }

    Ok(res)
}

fn transform_range_impl(
    tfa: &mut TransformFuncArg,
    running_fn: impl TransformFn,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = running_fn(tfa)?;
    set_last_values(&mut rvs);
    return Ok(rvs);
}

fn transform_range_avg(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, transform_running_avg)
}

fn transform_range_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, transform_running_max)
}

fn transform_range_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, transform_running_min)
}

fn transform_range_sum(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_range_impl(tfa, transform_running_sum)
}

fn transform_range_quantile(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let phi = get_scalar_float(tfa, 0, Some(0_f64))?;

    let mut series = get_series(tfa, 1)?;
    range_quantile(phi, &mut series);
    Ok(series)
}

fn transform_range_median(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    range_quantile(0.5, &mut series);
    Ok(series)
}

fn range_quantile(phi: f64, series: &mut Vec<Timeseries>) {
    let mut values = get_float64s(series.len()).to_vec();

    for ts in series.iter_mut() {
        let mut last_idx = 0;
        values.clear();
        for (i, v) in ts.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            values.push(*v);
            last_idx = i;
        }
        if last_idx > 0 {
            values.sort_by(|a, b| a.total_cmp(&b));
            ts.values[last_idx] = quantile_sorted(phi, &values)
        }
    }

    set_last_values(series);
}

fn transform_range_first(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        let len = ts.values.len();
        let first = get_first_non_nan_index(&ts.values);
        if first >= len - 1 {
            continue;
        }

        let v_first = ts.values[first];
        for v in ts.values[first..].iter_mut() {
            if !v.is_nan() {
                *v = v_first;
            }
        }
    }

    Ok(series)
}

fn transform_range_last(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    set_last_values(&mut series);
    Ok(series)
}

fn set_last_values(tss: &mut Vec<Timeseries>) {
    for ts in tss {
        let last = get_last_non_nan_index(&ts.values);
        if last == 0 {
            continue;
        }
        let v_last = ts.values[last];
        for v in ts.values[0..last].iter_mut() {
            if !v.is_nan() {
                *v = v_last;
            }
        }
    }
}

fn transform_smooth_exponential(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let sf = get_scalar_float(&tfa, 1, Some(1.0))?;
    let sf_val = if sf.is_nan() { 1.0 } else { sf.clamp(0.0, 1.0) };

    let mut series = get_series(tfa, 0)?;

    for ts in series.iter_mut() {
        let len = ts.values.len();

        // skip NaN and Inf
        let mut i = 0;
        for (j, v) in ts.values.iter().enumerate() {
            if v.is_finite() {
                i = j;
                continue;
            }
            break;
        }

        if i >= len {
            continue;
        }

        let mut avg = ts.values[0];
        i += 1;

        for value in ts.values[i..].iter_mut() {
            if !value.is_nan() {
                avg = avg * (1.0 - sf_val) + *value * sf_val;
                *value = avg;
            }
        }
    }

    Ok(series)
}

fn transform_remove_resets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        remove_counter_resets_maybe_nans(&mut ts.values);
    }
    Ok(series)
}

fn transform_union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() < 1 {
        return eval_number(&mut tfa.ec, NAN);
    }

    let series = get_series(tfa, 0)?;

    let len = series.len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);
    let mut m: HashSet<String> = HashSet::with_capacity(len);

    for arg in tfa.args.iter_mut().skip(1) {
        let other_series = arg.get_instant_vector(tfa.ec)?;
        for mut ts in other_series.into_iter() {
            // todo: get into a pre-allocated buffer
            let key = ts.metric_name.to_string();

            if m.insert(key) {
                rvs.push(std::mem::take(&mut ts));
            }
        }
    }

    return Ok(rvs);
}

fn transform_label_keep(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut keep_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let keep_label = get_string(&tfa, i)?;
        keep_labels.push(keep_label);
    }

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        ts.metric_name.remove_tags_on(&keep_labels)
    }

    Ok(series)
}

fn transform_label_del(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut del_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let del_label = get_string(&tfa, i)?;
        del_labels.push(del_label);
    }

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        ts.metric_name.remove_tags(&del_labels[0..])
    }

    Ok(series)
}

fn transform_label_set(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (dst_labels, dst_values) = get_string_pairs(tfa, 1)?;
    let mut series = get_series(tfa, 0)?;

    label_set(&mut series, &dst_labels, &dst_values);

    Ok(series)
}

fn transform_alias(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let alias = get_string(tfa, 1)?;
    let mut series = get_series(tfa, 0)?;

    label_set(&mut series, &[METRIC_NAME_LABEL.to_string()], &[alias]);

    Ok(series)
}

fn label_set(series: &mut Vec<Timeseries>, dst_labels: &[String], dst_values: &[String]) {
    for ts in series.iter_mut() {
        for (dst_label, value) in dst_labels.iter().zip(dst_values.iter()) {
            if value.len() == 0 {
                ts.metric_name.remove_tag(&dst_label);
            } else {
                ts.metric_name.set_tag(dst_label, value)
            }
        }
    }
}

fn transform_label_uppercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_value_func(tfa, |x| x.to_uppercase())
}

fn transform_label_lowercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_value_func(tfa, |x| x.to_lowercase())
}

fn transform_label_value_func(
    tfa: &mut TransformFuncArg,
    f: fn(arg: &str) -> String,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels = Vec::with_capacity(tfa.args.len() - 1);
    for i in 1..tfa.args.len() {
        let label = get_string(&tfa, i)?;
        labels.push(label);
    }
    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        for label in labels.iter() {
            let dst_value = get_tag_value(&mut ts.metric_name, label);
            let transformed = &*f(&dst_value);

            if transformed.len() == 0 {
                ts.metric_name.remove_tag(label);
            } else {
                ts.metric_name.set_tag(label, transformed);
            }
        }
    }

    Ok(series)
}

fn transform_label_map(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label = get_label(tfa, "", 1)?;

    let (src_values, dst_values) = get_string_pairs(tfa, 2)?;
    let mut m: HashMap<&str, &str> = HashMap::with_capacity(src_values.len());
    for (i, src_value) in src_values.iter().enumerate() {
        m.insert(src_value, &dst_values[i]);
    }

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        let mut dst_value = get_tag_value(&mut ts.metric_name, &label);
        if let Some(value) = m.get(dst_value.as_str()) {
            dst_value.push_str(value);
        }
        if dst_value.len() == 0 {
            ts.metric_name.remove_tag(&label);
        } else {
            ts.metric_name.set_tag(&label, &dst_value);
        }
    }

    Ok(series)
}

fn transform_drop_common_labels(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    for i in 1..tfa.args.len() {
        let mut other = get_series(tfa, i)?;
        series.append(&mut other);
    }

    let mut counts_map: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for ts in series.iter() {
        ts.metric_name.count_label_values(&mut counts_map);
    }

    let series_len = series.len();
    // m.iter().filter(|entry| entry.1)
    for (label_name, x) in counts_map.iter() {
        for (_, count) in x {
            if *count != series_len {
                continue;
            }
            for ts in series.iter_mut() {
                ts.metric_name.remove_tag(label_name);
            }
        }
    }

    Ok(series)
}

fn transform_label_copy(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, false)
}

fn transform_label_move(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, true)
}

fn transform_label_copy_ext(
    tfa: &mut TransformFuncArg,
    remove_src_labels: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let (src_labels, dst_labels) = get_string_pairs(tfa, 1)?;

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        for (src_label, dst_label) in src_labels.iter().zip(dst_labels.iter()) {
            let value = ts.metric_name.get_tag_value(src_label);
            if value.is_none() {
                continue;
            }
            let value = value.unwrap();
            if value.len() == 0 {
                // do not remove destination label if the source label doesn't exist.
                continue;
            }

            // this is done because value is a live (owned) ref to data in ts.metric_name
            // because of this, there was an outstanding borrow
            let v = value.clone();
            ts.metric_name.set_tag(dst_label, &v);

            if remove_src_labels && src_label != dst_label {
                ts.metric_name.remove_tag(src_label)
            }
        }
    }

    Ok(series)
}

fn get_arg<'a>(tfa: &'a TransformFuncArg, index: usize) -> RuntimeResult<&'a QueryValue> {
    if index >= tfa.args.len() {
        return Err(RuntimeError::ArgumentError(format!(
            "expected at least {} args; got {}",
            index + 1,
            tfa.args.len()
        )));
    }
    Ok(&tfa.args[index])
}

fn get_string_pairs(
    tfa: &mut TransformFuncArg,
    start: usize,
) -> RuntimeResult<(Vec<String>, Vec<String>)> {
    if start >= tfa.args.len() {
        return Err(RuntimeError::ArgumentError(format!(
            "expected at least {} args; got {}",
            start + 2,
            tfa.args.len()
        )));
    }
    let args = &tfa.args[start..];
    let arg_len = args.len();
    if arg_len % 2 != 0 {
        return Err(RuntimeError::ArgumentError(format!(
            "the number of string args must be even; got {arg_len}"
        )));
    }
    let result_len = arg_len / 2;
    let mut ks: Vec<String> = Vec::with_capacity(result_len);
    let mut vs: Vec<String> = Vec::with_capacity(result_len);
    let mut i = start;
    while i < arg_len {
        let k = get_string(&tfa, i)?;
        ks.push(k);
        let v = get_string(&tfa, i + 1)?;
        vs.push(v);
        i += 2;
    }

    Ok((ks, vs))
}

fn transform_label_join(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = get_string(&tfa, 1)?;
    let separator = get_string(&tfa, 2)?;

    // todo: user something like SmallVec/StaticVec/ArrayVec
    let mut src_labels: Vec<String> = Vec::with_capacity(tfa.args.len() - 3);
    for i in 3..tfa.args.len() {
        let src_label = get_string(&tfa, i)?;
        src_labels.push(src_label);
    }

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        let mut dst_value = get_tag_value(&mut ts.metric_name, &dst_label);
        // use some manner of string buffer

        dst_value.clear(); //??? test this

        for (j, src_label) in src_labels.iter().enumerate() {
            if let Some(src_value) = ts.metric_name.get_tag_value(src_label) {
                dst_value.push_str(&src_value);
            }
            if j + 1 < src_labels.len() {
                dst_value.push_str(&separator)
            }
        }

        if dst_value.len() == 0 {
            ts.metric_name.remove_tag(&dst_label);
        } else {
            ts.metric_name.set_tag(&dst_label, &dst_value);
        }
    }

    Ok(series)
}

fn transform_label_transform(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label = get_string(&tfa, 1)?;
    let regex = get_string(&tfa, 2)?;
    let replacement = get_string(&tfa, 3)?;

    // todo: would it be useful to use a cache ?
    let r = compile_regexp(&regex);
    match r {
        Ok(..) => {}
        Err(err) => {
            return Err(RuntimeError::from(format!(
                "cannot compile regex {}: {:?}",
                &regex, err,
            )));
        }
    }

    let mut series = get_series(tfa, 0)?;
    label_replace(&mut series, &label, &r.unwrap(), &label, &replacement)
}

fn transform_label_replace(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let regex = get_string(tfa, 4)?;

    process_anchored_regex(tfa, regex.as_str(), |tfa, r| {
        let dst_label = get_string(tfa, 1)?;
        let replacement = get_string(tfa, 2)?;
        let src_label = get_string(tfa, 3)?;
        let mut series = get_series(tfa, 0)?;

        label_replace(&mut series, &src_label, &r, &dst_label, &replacement)
    })
}

fn label_replace(
    tss: &mut Vec<Timeseries>,
    src_label: &str,
    r: &Regex,
    dst_label: &str,
    replacement: &str,
) -> RuntimeResult<Vec<Timeseries>> {
    for ts in tss.iter_mut() {
        let src_value = ts.metric_name.get_tag_value(src_label);
        if src_value.is_none() {
            continue;
        }
        let src_value = src_value.unwrap();
        if !r.is_match(&src_value) {
            continue;
        }
        let b = r.replace_all(&src_value, replacement);
        if b.len() == 0 {
            ts.metric_name.remove_tag(dst_label)
        } else {
            // if we have borrowed src_value, we need to clone it to avoid holding
            // a borrowed ref to ts.metric_name
            match b {
                Cow::Borrowed(_) => {
                    let cloned = src_value.clone();
                    ts.metric_name.set_tag(dst_label, &cloned);
                }
                Cow::Owned(owned) => {
                    ts.metric_name.set_tag(dst_label, &owned);
                }
            };
        }
    }

    Ok(std::mem::take(tss))
}

fn transform_label_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_name = get_string(tfa, 1)?;
    let mut x: f64;

    let mut series = get_series(tfa, 0)?;
    for ts in series.iter_mut() {
        ts.metric_name.reset_metric_group();
        if let Some(label_value) = ts.metric_name.get_tag_value(&label_name) {
            x = match label_value.parse::<f64>() {
                Ok(v) => v,
                Err(..) => NAN,
            };

            for val in ts.values.iter_mut() {
                *val = x;
            }
        }
    }

    // do not remove timeseries with only NaN values, so `default` could be applied to them:
    // label_value(q, "label") default 123
    Ok(std::mem::take(&mut series))
}

#[inline]
fn process_anchored_regex<F>(
    tfa: &mut TransformFuncArg,
    re: &str,
    handler: F,
) -> RuntimeResult<Vec<Timeseries>>
where
    F: Fn(&mut TransformFuncArg, &Regex) -> RuntimeResult<Vec<Timeseries>>,
{
    let anchored = format!("^(?:{})$", re);
    match Regex::new(&anchored) {
        Err(e) => Err(RuntimeError::from(format!(
            "cannot compile regexp {} : {}",
            &re, e
        ))),
        Ok(ref r) => handler(tfa, r),
    }
}

fn transform_label_match(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_name = get_label(tfa, "", 1)?;
    let label_re = get_label(tfa, "regexp", 1)?;

    process_anchored_regex(tfa, &label_re, move |tfa, r| {
        let mut series = get_series(tfa, 0)?;
        series.retain(|ts| {
            if let Some(label_value) = ts.metric_name.get_tag_value(&label_name) {
                r.is_match(label_value)
            } else {
                false
            }
        });

        Ok(series)
    })
}

fn transform_label_mismatch(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_re = get_label(tfa, "regexp", 2)?;

    process_anchored_regex(tfa, &label_re, |tfa, r| {
        let label_name = get_label(tfa, "", 1)?;
        let mut series = get_series(tfa, 0)?;
        series.retain(|ts| {
            if let Some(label_value) = ts.metric_name.get_tag_value(&label_name) {
                !r.is_match(label_value)
            } else {
                false
            }
        });

        Ok(std::mem::take(&mut series))
    })
}

fn transform_label_graphite_group(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut group_ids: Vec<i64> = Vec::with_capacity(tfa.args.len() - 1);
    for i in 1..tfa.args.len() {
        match tfa.args[i + 1].get_int() {
            Ok(gid) => group_ids.push(gid),
            Err(e) => {
                let msg = format!("cannot get group name from arg #{}: {:?}", i + 1, e);
                return Err(RuntimeError::ArgumentError(msg));
            }
        }
    }

    let mut series = get_series(tfa, 0)?;

    for ts in series.iter_mut() {
        let groups: Vec<&str> = ts.metric_name.metric_group.split(DOT_SEPARATOR).collect();
        let mut group_name: String =
            String::with_capacity(ts.metric_name.metric_group.len() + groups.len());
        for (j, group_id) in group_ids.iter().enumerate() {
            if *group_id >= 0_i64 && *group_id < (groups.len() as i64) {
                let idx = *group_id as usize;
                group_name.push_str(groups[idx]);
            }
            if j < group_ids.len() - 1 {
                group_name.push('.')
            }
        }
        ts.metric_name.metric_group = group_name
    }

    Ok(std::mem::take(&mut series))
}

const DOT_SEPARATOR: &str = ".";

fn transform_limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let limit: usize;
    let offset: usize;

    match tfa.args[0].get_int() {
        Err(e) => {
            return Err(RuntimeError::from(format!(
                "cannot obtain limit arg: {:?}",
                e
            )));
        }
        Ok(l) => {
            limit = l as usize;
        }
    }

    match get_int_number(&tfa, 1) {
        Err(_) => {
            return Err(RuntimeError::from("cannot obtain offset arg"));
        }
        Ok(v) => {
            offset = v as usize;
        }
    }

    let mut rvs = get_series(tfa, 2)?;

    // remove_empty_series so offset will be calculated after empty series
    // were filtered out.
    remove_empty_series(&mut rvs);

    if rvs.len() >= offset {
        rvs.drain(0..offset);
    }
    if rvs.len() > limit {
        rvs.resize(limit, Timeseries::default());
    }

    Ok(std::mem::take(&mut rvs))
}

fn transform_round(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args_len = tfa.args.len();

    if args_len < 1 || args_len > 2 {
        return Err(RuntimeError::ArgumentError(format!(
            "unexpected number of arguments: #{}; want 1 or 2",
            tfa.args.len()
        )));
    }

    let nearest = if args_len == 1 {
        let len = tfa.ec.data_points();
        vec![1_f64; len]
    } else {
        get_scalar(tfa, 1)?
    };

    let tf = move |values: &mut [f64]| {
        let mut n_prev: f64 = values[0];
        let mut p10: f64 = 0.0;
        for (v, n) in values.iter_mut().zip(nearest.iter()) {
            if *n != n_prev {
                n_prev = *n;
                let (_, e) = from_float(*n);
                p10 = -(e as f64).powi(10);
            }
            *v += 0.5 * copysign(*n, *v);
            *v -= fmod(*v, *n);
            let (x, _) = modf(*v * p10);
            *v = x / p10;
        }
    };

    transform_series(tfa, tf)
}

fn transform_sgn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        for v in values {
            *v = v.signum();
        }
    };

    transform_series(tfa, tf)
}

fn transform_scalar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let arg = get_arg(tfa, 0)?;
    match arg {
        // Verify whether the arg is a string.
        // Then try converting the string to number.
        QueryValue::String(s) => {
            let n = parse_number(&s).map_or_else(|_| f64::NAN, |n| n);
            eval_number(&mut tfa.ec, n)
        }
        QueryValue::Scalar(f) => eval_number(&tfa.ec, *f),
        _ => {
            // The arg isn't a string. Extract scalar from it.
            if tfa.args.len() != 1 {
                eval_number(&tfa.ec, NAN)
            } else {
                let mut arg = arg.get_instant_vector(tfa.ec)?.remove(0);
                arg.metric_name.reset();
                Ok(vec![arg])
            }
        }
    }
}

fn sort_by_label_impl(tfa: &mut TransformFuncArg, is_desc: bool) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = Vec::with_capacity(1);
    let mut series = get_series(tfa, 0)?;

    for arg in tfa.args.iter().skip(1) {
        let label = arg.get_string()?;
        labels.push(label);
    }

    series.sort_by(|first, second| {
        for label in labels.iter() {
            let a = first.metric_name.get_tag_value(&label);
            let b = second.metric_name.get_tag_value(&label);
            match (a, b) {
                (Some(a1), Some(b1)) => {
                    if a1 == b1 {
                        continue;
                    } else if is_desc {
                        return b1.cmp(&a1);
                    } else {
                        return a1.cmp(&b1);
                    }
                }
                (Some(_), None) => return Ordering::Greater,
                (None, Some(_)) => return Ordering::Less,
                (None, None) => return Ordering::Equal,
            }
        }
        return Ordering::Equal;
    });

    Ok(series)
}

fn transform_sort_by_label(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, false)
}

fn transform_sort_by_label_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, true)
}

fn label_alpha_numeric_sort_impl(
    tfa: &mut TransformFuncArg,
    is_desc: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = vec![];
    for (i, arg) in tfa.args.iter().skip(1).enumerate() {
        let label = arg.get_string().map_err(|err| {
            RuntimeError::ArgumentError(format!(
                "cannot parse label {} for sorting: {:?}",
                i + 1,
                err
            ))
        })?;
        labels.push(label);
    }

    let mut res = get_series(tfa, 0)?;
    res.sort_by(|first, second| {
        for label in &labels {
            match (
                first.metric_name.get_tag_value(&label),
                second.metric_name.get_tag_value(&label),
            ) {
                (None, None) => continue,
                (Some(a), Some(b)) => {
                    return if is_desc {
                        compare_str_alphanumeric(b, a)
                    } else {
                        compare_str_alphanumeric(a, b)
                    };
                }
                (None, Some(_)) => {
                    return if is_desc {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
                (Some(_), None) => {
                    return if is_desc {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
            }
        }
        Ordering::Equal
    });

    Ok(res)
}

/// Compare two strings.
/// Source: https://crates.io/crates/alphanumeric-sort
pub fn compare_str_alphanumeric<A: AsRef<str>, B: AsRef<str>>(a: A, b: B) -> Ordering {
    let mut c1 = a.as_ref().chars();
    let mut c2 = b.as_ref().chars();

    // this flag is to handle something like "1點" < "1-1點"
    let mut last_is_number = false;

    let mut v1: Option<char> = None;
    let mut v2: Option<char> = None;

    loop {
        let ca = {
            match v1.take() {
                Some(c) => c,
                None => match c1.next() {
                    Some(c) => c,
                    None => {
                        return if v2.take().is_some() || c2.next().is_some() {
                            Ordering::Less
                        } else {
                            Ordering::Equal
                        };
                    }
                },
            }
        };

        let cb = {
            match v2.take() {
                Some(c) => c,
                None => match c2.next() {
                    Some(c) => c,
                    None => {
                        return Ordering::Greater;
                    }
                },
            }
        };

        let b_zero = f64::from(b'0');

        if ('0'..='9').contains(&ca) && ('0'..='9').contains(&cb) {
            let mut da = f64::from(ca as u32) - b_zero;
            let mut db = f64::from(cb as u32) - b_zero;

            // this counter is to handle something like "001" > "01"
            let mut dc = 0isize;

            for ca in c1.by_ref() {
                if ('0'..='9').contains(&ca) {
                    da = da * 10.0 + (f64::from(ca as u32) - b_zero);
                    dc += 1;
                } else {
                    v1 = Some(ca);
                    break;
                }
            }

            for cb in c2.by_ref() {
                if ('0'..='9').contains(&cb) {
                    db = db * 10.0 + (f64::from(cb as u32) - b_zero);
                    dc -= 1;
                } else {
                    v2 = Some(cb);
                    break;
                }
            }

            last_is_number = true;

            let ordering = da.total_cmp(&db);
            if ordering != Ordering::Equal {
                return ordering;
            } else {
                match dc.cmp(&0) {
                    Ordering::Equal => (),
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Less => return Ordering::Less,
                }
            }
        } else {
            match ca.cmp(&cb) {
                Ordering::Equal => last_is_number = false,
                Ordering::Greater => {
                    return if last_is_number && (ca > (255 as char)) ^ (cb > (255 as char)) {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
                Ordering::Less => {
                    return if last_is_number && (ca > (255 as char)) ^ (cb > (255 as char)) {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
            }
        }
    }
}

fn transform_sort_impl(
    tfa: &mut TransformFuncArg,
    is_desc: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series(tfa, 0)?;
    series.sort_by(move |first, second| {
        let a = &first.values;
        let b = &second.values;
        let mut n = a.len() - 1;
        loop {
            if !a[n].is_nan() && !b[n].is_nan() && a[n] != b[n] {
                break;
            }
            if n == 0 {
                return Ordering::Greater;
            }
            n -= 1;
        }

        if is_desc {
            return b[n].total_cmp(&a[n]);
        }
        return a[n].total_cmp(&b[n]);
    });

    Ok(std::mem::take(&mut series))
}

trait RandFunc: FnMut() -> f64 {}
impl<T> RandFunc for T where T: FnMut() -> f64 {}

fn create_rng(tfa: &mut TransformFuncArg) -> RuntimeResult<StdRng> {
    if tfa.args.len() == 1 {
        return match tfa.args[0].get_int() {
            Err(e) => Err(e),
            Ok(val) => match u64::try_from(val) {
                Err(_) => Err(RuntimeError::ArgumentError(
                    format!("invalid rand seed {}", val).to_string(),
                )),
                Ok(seed) => Ok(StdRng::seed_from_u64(seed)),
            },
        };
    }
    match StdRng::from_rng(thread_rng()) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(
                format!("Error constructing rng {:?}", e).to_string(),
            ))
        }
        Ok(rng) => Ok(rng),
    }
}

macro_rules! create_rand_func {
    ($name: ident, $f:expr) => {
        fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            let mut rng: StdRng = create_rng(tfa)?;
            let mut tss = eval_number(&tfa.ec, 0.0)?;
            for value in tss[0].values.iter_mut() {
                *value = $f(&mut rng);
            }
            Ok(tss)
        }
    };
}

create_rand_func!(transform_rand, |r: &mut StdRng| r.gen::<f64>());

create_rand_func!(transform_rand_norm, |r: &mut StdRng| {
    <StandardNormal as Distribution<f64>>::sample::<StdRng>(&StandardNormal, r) as f64
});

create_rand_func!(transform_rand_exp, |r: &mut StdRng| {
    <Exp1 as Distribution<f64>>::sample::<StdRng>(&Exp1, r) as f64
});

fn transform_pi(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_number(&tfa.ec, f64::PI())
}

fn transform_now(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let now: f64 = Utc::now().timestamp() as f64 / 1e9_f64;
    eval_number(&tfa.ec, now)
}

#[inline]
fn bitmap_and(a: u64, b: u64) -> u64 {
    a & b
}

#[inline]
fn bitmap_or(a: u64, b: u64) -> u64 {
    a | b
}

#[inline]
fn bitmap_xor(a: u64, b: u64) -> u64 {
    a ^ b
}

fn transform_bitmap_and(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_and)
}

fn transform_bitmap_or(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_or)
}

fn transform_bitmap_xor(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_xor)
}

fn transform_bitmap_impl(
    tfa: &mut TransformFuncArg,
    bitmap_func: fn(a: u64, b: u64) -> u64,
) -> RuntimeResult<Vec<Timeseries>> {
    let mask = get_scalar_float(&tfa, 1, None)? as u64;

    let tf = |values: &mut [f64]| {
        for v in values.iter_mut() {
            *v = bitmap_func(*v as u64, mask) as f64;
        }
    };

    transform_series(tfa, tf)
}

fn parse_zone(tz_name: &str) -> RuntimeResult<Tz> {
    match tz_name.parse() {
        Ok(zone) => Ok(zone),
        Err(e) => Err(RuntimeError::ArgumentError(format!(
            "unable to parse tz: {:?}",
            e
        ))),
    }
}

fn transform_timezone_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tz_name = match get_string(tfa, 0) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot get timezone name from arg: {:?}",
                e
            )))
        }
        Ok(s) => s,
    };

    let zone = match parse_zone(&tz_name) {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot load timezone {tz_name}: {:?}",
                e
            )))
        }
        Ok(res) => res,
    };

    let timestamps = tfa.ec.get_timestamps()?;
    let mut values: Vec<f64> = Vec::with_capacity(timestamps.len());
    for v in timestamps.iter() {
        // todo(perf) construct a DateTime in the tz and update timestamp
        let ofs = if let Some(val) = get_timezone_offset(&zone, *v) {
            val as f64
        } else {
            f64::NAN
        };
        values.push(ofs);
    }
    let ts = Timeseries {
        metric_name: MetricName::default(),
        values,
        timestamps: timestamps.clone(),
    };

    Ok(vec![ts])
}

fn transform_time(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_time(&mut tfa.ec)
}

fn transform_vector(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series(tfa, 0)?;
    Ok(series)
}

fn transform_step(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.step as f64 / 1e3_f64;
    eval_number(&tfa.ec, v)
}

#[inline]
fn transform_start(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.start as f64 / 1e3_f64;
    eval_number(&tfa.ec, v)
}

#[inline]
fn transform_end(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.end as f64 / 1e3_f64;
    eval_number(&tfa.ec, v)
}

/// copy_timeseries returns a copy of tss.
fn copy_timeseries(tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for src in tss {
        rvs.push(src.clone());
    }
    return rvs;
}

fn get_tag_value(mn: &MetricName, dst_label: &str) -> String {
    match mn.get_tag_value(dst_label) {
        Some(val) => val.to_owned(),
        None => "".to_string(),
    }
}

fn remove_counter_resets_maybe_nans(values: &mut Vec<f64>) {
    let mut i = 0;
    while i < values.len() && values[i].is_nan() {
        i += 1;
    }
    if i > values.len() {
        values.clear();
        return;
    }
    if i > 0 {
        if i == 1 {
            values.remove(i);
        } else {
            values.drain(0..i);
        }
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    for v in values.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let d = *v - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely jitter from `Prometheus HA pairs`.
                // Just substitute v with prev_value.
                *v = prev_value;
            } else {
                correction += prev_value
            }
        }
        prev_value = *v;
        *v = *v + correction;
    }
}

fn get_string_value(arg: &QueryValue, arg_num: usize) -> RuntimeResult<String> {
    let res = match arg {
        QueryValue::String(s) => Ok(s.clone()), // todo: use .into ??
        QueryValue::Scalar(f) => Ok(f.to_string()),
        QueryValue::InstantVector(series) => {
            if series.len() != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    series.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            for v in series[0].values.iter() {
                if !v.is_nan() {
                    let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
                    return Err(RuntimeError::ArgumentError(msg));
                }
            }
            // todo: return reference
            return Ok(series[0].metric_name.metric_group.clone());
        }
        _ => Err(RuntimeError::ArgumentError(
            "string parameter expected ".to_string(),
        )),
    };
    res
}

fn get_string(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<String> {
    if let Some(arg) = tfa.args.get(arg_num) {
        return get_string_value(arg, arg_num);
    }
    let msg = format!("missing arg # {}", arg_num + 1);
    return Err(RuntimeError::ArgumentError(msg));
}

fn get_label(tfa: &TransformFuncArg, name: &str, arg_num: usize) -> RuntimeResult<String> {
    if let Some(arg) = tfa.args.get(arg_num) {
        return get_string_value(arg, arg_num);
    }
    let msg = format!("cannot read {} label name", name);
    return Err(RuntimeError::ArgumentError(msg));
}

pub fn get_series(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<Vec<Timeseries>> {
    if let Some(arg) = tfa.args.get(arg_num) {
        return arg.get_instant_vector(tfa.ec);
    }
    let msg = format!("missing series arg # {}", arg_num + 1);
    return Err(RuntimeError::ArgumentError(msg));
}

// TODO: COW, or return Iterator
pub fn get_scalar(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<Vec<f64>> {
    // todo: check bounds
    let arg = tfa.args.get(arg_num);
    if arg.is_none() {
        let msg = format!("missing scalar arg # {}", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    let arg = arg.unwrap();
    match arg {
        QueryValue::Scalar(val) => {
            // todo: use object pool here
            let len = tfa.ec.data_points();
            // todo: tinyvec
            let values = vec![*val; len];
            Ok(values)
        }
        QueryValue::InstantVector(s) => {
            if s.len() != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    s.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            Ok(s[0].values.clone())
        }
        _ => {
            let msg = format!(
                "arg # {} expected float or a single timeseries; got {}",
                arg_num + 1,
                arg.data_type()
            );
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
}

fn get_scalar_float(
    tfa: &TransformFuncArg,
    arg_num: usize,
    default_value: Option<f64>,
) -> RuntimeResult<f64> {
    // todo: check bounds
    let arg = &tfa.args[arg_num];
    match arg {
        QueryValue::Scalar(val) => return Ok(*val),
        QueryValue::InstantVector(s) => {
            let len = s.len();
            if len == 0 {
                if let Some(default) = default_value {
                    return Ok(default);
                }
            }
            if len != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    s.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            return Ok(s[0].values[0]);
        }
        _ => {}
    }

    let msg = format!(
        "arg # {} expected float or a single timeseries; got {}",
        arg_num + 1,
        arg.data_type()
    );
    return Err(RuntimeError::ArgumentError(msg));
}

pub(crate) fn get_int_number(tfa: &TransformFuncArg, arg_num: usize) -> RuntimeResult<i64> {
    get_scalar_float(tfa, arg_num, Some(0_f64)).map(float_to_int_bounded)
}

fn expect_transform_args_num(tfa: &TransformFuncArg, expected: usize) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    return Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {arg_count}; want {expected}"
    )));
}
