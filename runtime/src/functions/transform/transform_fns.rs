use std::default::Default;

use metricsql::ast::FunctionExpr;
use metricsql::functions::TransformFunction;

use crate::eval::EvalConfig;
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::absent::transform_absent;
use crate::functions::transform::bitmap::{
    transform_bitmap_and, transform_bitmap_or, transform_bitmap_xor,
};
use crate::functions::transform::clamp::{
    transform_clamp, transform_clamp_max, transform_clamp_min,
};
use crate::functions::transform::datetime::{
    transform_day_of_month, transform_day_of_week, transform_days_in_month, transform_hour,
    transform_minute, transform_month, transform_now, transform_time, transform_timezone_offset,
    transform_year,
};
use crate::functions::transform::end::transform_end;
use crate::functions::transform::histogram::{
    transform_buckets_limit, transform_histogram_avg, transform_histogram_quantile,
    transform_histogram_quantiles, transform_histogram_share, transform_histogram_stddev,
    transform_histogram_stdvar, transform_prometheus_buckets,
};
use crate::functions::transform::interpolate::transform_interpolate;
use crate::functions::transform::keep_last_value::transform_keep_last_value;
use crate::functions::transform::keep_next_value::transform_keep_next_value;
use crate::functions::transform::labels::{
    transform_alias, transform_drop_common_labels, transform_label_copy, transform_label_del,
    transform_label_graphite_group, transform_label_join, transform_label_keep,
    transform_label_lowercase, transform_label_map, transform_label_match,
    transform_label_mismatch, transform_label_move, transform_label_replace, transform_label_set,
    transform_label_transform, transform_label_uppercase, transform_label_value,
};
use crate::functions::transform::limit_offset::transform_limit_offset;
use crate::functions::transform::math::{
    transform_abs, transform_acos, transform_acosh, transform_asin, transform_asinh,
    transform_atan, transform_atanh, transform_ceil, transform_cos, transform_cosh, transform_deg,
    transform_exp, transform_floor, transform_ln, transform_log10, transform_log2, transform_pi,
    transform_rad, transform_round, transform_sgn, transform_sin, transform_sinh, transform_sqrt,
    transform_tan, transform_tanh,
};
use crate::functions::transform::random::{
    transform_rand, transform_rand_exp, transform_rand_norm,
};
use crate::functions::transform::range::{
    transform_range_avg, transform_range_first, transform_range_last,
    transform_range_linear_regression, transform_range_max, transform_range_median,
    transform_range_min, transform_range_normalize, transform_range_quantile, transform_range_ru,
    transform_range_stddev, transform_range_stdvar, transform_range_sum,
    transform_range_trim_outliers, transform_range_trim_spikes, transform_range_trim_zscore,
    transform_range_zscore,
};
use crate::functions::transform::remove_resets::transform_remove_resets;
use crate::functions::transform::running::{
    transform_running_avg, transform_running_max, transform_running_min, transform_running_sum,
};
use crate::functions::transform::scalar::transform_scalar;
use crate::functions::transform::smooth_exponential::transform_smooth_exponential;
use crate::functions::transform::sort::{
    transform_sort, transform_sort_alpha_numeric, transform_sort_alpha_numeric_desc,
    transform_sort_by_label, transform_sort_by_label_desc, transform_sort_desc,
};
use crate::functions::transform::start::transform_start;
use crate::functions::transform::step::transform_step;
use crate::functions::transform::union::transform_union;
use crate::functions::transform::vector::transform_vector;
use crate::runtime_error::RuntimeResult;
use crate::{QueryValue, Timeseries};

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

pub(crate) trait TransformValuesFn: FnMut(&mut [f64]) -> () {}
impl<T> TransformValuesFn for T where T: FnMut(&mut [f64]) -> () {}

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

pub(crate) fn transform_series(
    tfa: &mut TransformFuncArg,
    tf: impl TransformValuesFn,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    do_transform_values(&mut series, tf, tfa.keep_metric_names)
}

pub(super) fn do_transform_values(
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
