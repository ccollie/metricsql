pub(crate) use histogram::vmrange_buckets_to_le;
use metricsql_parser::functions::{BuiltinFunction, TransformFunction};
pub(crate) use utils::{extract_labels_from_expr, get_timezone_offset};

use crate::execution::EvalConfig;
use crate::functions::transform::math::{
    abs, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, deg, exp, floor, ln, log10, log2,
    rad, sgn, sin, sinh, sqrt, tan, tanh, transform_pi,
};
use crate::functions::transform::ru::transform_ru;
use crate::runtime_error::RuntimeResult;
use crate::types::{FunctionArgs, Timeseries};
use absent::transform_absent;
use bitmap::{transform_bitmap_and, transform_bitmap_or, transform_bitmap_xor};
use clamp::{clamp, clamp_max, clamp_min};
use datetime::{
    day_of_month, day_of_week, day_of_year, days_in_month, hour, minute, month, now, time,
    timezone_offset, year,
};
use drop_empty_series::transform_drop_empty_series;
use end::transform_end;
use histogram::{
    buckets_limit, histogram_avg, histogram_quantile, histogram_quantiles, histogram_share,
    histogram_stddev, histogram_stdvar, prometheus_buckets,
};
use interpolate::interpolate;
use keep_last_value::keep_last_value;
use keep_next_value::keep_next_value;
use labels::{
    alias, drop_common_labels, label_copy, label_del, label_graphite_group, label_join, label_keep,
    label_lowercase, label_map, label_match, label_mismatch, label_move, label_replace, label_set,
    label_transform, label_uppercase, label_value, labels_equal,
};
use limit_offset::limit_offset;
use metricsql_parser::ast::FunctionExpr;
use rand::{rand, rand_exp, rand_norm};
use range::{
    range_avg, range_first, range_last, range_linear_regression, range_mad, range_max,
    range_median, range_min, range_normalize, range_stddev, range_stdvar, range_sum,
    range_trim_outliers, range_trim_spikes, range_trim_zscore, range_zscore,
    transform_range_quantile,
};
use remove_resets::remove_resets;
use round::round;
use running::{running_avg, running_max, running_min, running_sum};
use scalar::scalar;
use smooth_exponential::smooth_exponential;
use sort::{
    sort, sort_alpha_numeric, sort_alpha_numeric_desc, sort_by_label, sort_by_label_desc, sort_desc,
};
use start::transform_start;
use step::step;
use union::union;
use vector::vector;

use crate::prelude::QueryValue;
use crate::RuntimeError;
pub(crate) use union::handle_union;

mod absent;
mod bitmap;
mod clamp;
mod datetime;
mod drop_empty_series;
mod end;
mod histogram;
mod interpolate;
mod keep_last_value;
mod keep_next_value;
mod labels;
mod limit_offset;
mod math;
mod rand;
mod range;
mod remove_resets;
mod round;
mod ru;
mod running;
mod scalar;
mod smooth_exponential;
mod sort;
mod start;
mod step;
#[cfg(test)]
mod transform_test;
mod union;
mod utils;
mod vector;

pub(crate) struct TransformFuncArg<'a> {
    pub ec: &'a EvalConfig,
    pub fe: &'a FunctionExpr,
    pub args: FunctionArgs,
}

impl<'a> TransformFuncArg<'a> {
    fn function_name(&self) -> &str {
        self.fe.function.name()
    }

    fn get_arg(&self, arg_num: usize, name: &str) -> RuntimeResult<&QueryValue> {
        let Some(arg) = self.args.get(arg_num) else {
            let msg = format!(
                "error getting {name} arg calling {}()",
                self.fe.function.name()
            );
            return Err(RuntimeError::ArgumentError(msg));
        };
        Ok(arg)
    }

    /// Get a series argument by its index (takes ownership of the argument).
    pub(super) fn get_param_series(&mut self, arg_num: usize) -> RuntimeResult<Vec<Timeseries>> {
        if let Some(arg) = self.args.get_mut(arg_num) {
            let arg = std::mem::take(arg);
            return arg.into_instant_vector(self.ec);
        }
        let msg = format!("missing series arg # {}", arg_num + 1);
        Err(RuntimeError::ArgumentError(msg))
    }

    pub(super) fn get_param_scalar(
        &self,
        arg_num: usize,
        name: &str,
        default_value: Option<f64>,
    ) -> RuntimeResult<f64> {
        // todo: check bounds
        let arg = self.get_arg(arg_num, name)?;
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
            "expected a scalar {name} value for arg # {} calling {}; got a {}",
            arg_num + 1,
            self.fe.function,
            arg.data_type()
        );
        Err(RuntimeError::ArgumentError(msg))
    }

    pub(super) fn get_param_usize(&self, arg_num: usize, name: &str) -> RuntimeResult<usize> {
        let value = self.get_param_scalar(arg_num, name, None)?;
        if value < 0.0 {
            return Err(RuntimeError::ArgumentError(format!(
                "{name} must be a non-negative integer; got {value}"
            )));
        }
        Ok(value as usize)
    }

    pub(super) fn take_param_string(
        &mut self,
        arg_num: usize,
        name: &str,
    ) -> RuntimeResult<String> {
        if let Some(arg) = self.args.get_mut(arg_num) {
            let arg = std::mem::take(arg);
            if let QueryValue::String(s) = arg {
                return Ok(s);
            }

            if let Ok(s) = arg.get_string() {
                return Ok(s);
            }
        }
        Err(RuntimeError::ArgumentError(format!(
            "error getting string argument {name} calling function {}",
            self.function_name()
        )))
    }
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

pub(crate) trait TransformValuesFn: FnMut(&mut [f64]) {}

impl<T> TransformValuesFn for T where T: FnMut(&mut [f64]) {}

const fn get_transform_func(f: TransformFunction) -> TransformFuncHandler {
    use TransformFunction::*;
    match f {
        Abs => abs,
        Absent => transform_absent,
        Acos => acos,
        Acosh => acosh,
        Alias => alias,
        Asin => asin,
        Asinh => asinh,
        Atan => atan,
        Atanh => atanh,
        BitmapAnd => transform_bitmap_and,
        BitmapOr => transform_bitmap_or,
        BitmapXor => transform_bitmap_xor,
        BucketsLimit => buckets_limit,
        Ceil => ceil,
        Clamp => clamp,
        ClampMax => clamp_max,
        ClampMin => clamp_min,
        Cos => cos,
        Cosh => cosh,
        DayOfMonth => day_of_month,
        DayOfWeek => day_of_week,
        DayOfYear => day_of_year,
        DaysInMonth => days_in_month,
        Deg => deg,
        DropCommonLabels => drop_common_labels,
        DropEmptySeries => transform_drop_empty_series,
        End => transform_end,
        Exp => exp,
        Floor => floor,
        HistogramAvg => histogram_avg,
        HistogramQuantile => histogram_quantile,
        HistogramQuantiles => histogram_quantiles,
        HistogramShare => histogram_share,
        HistogramStddev => histogram_stddev,
        HistogramStdvar => histogram_stdvar,
        Hour => hour,
        Interpolate => interpolate,
        KeepLastValue => keep_last_value,
        KeepNextValue => keep_next_value,
        LabelCopy => label_copy,
        LabelDel => label_del,
        LabelGraphiteGroup => label_graphite_group,
        LabelJoin => label_join,
        LabelKeep => label_keep,
        LabelLowercase => label_lowercase,
        LabelMap => label_map,
        LabelMatch => label_match,
        LabelMismatch => label_mismatch,
        LabelMove => label_move,
        LabelReplace => label_replace,
        LabelsEqual => labels_equal,
        LabelSet => label_set,
        LabelTransform => label_transform,
        LabelUppercase => label_uppercase,
        LabelValue => label_value,
        LimitOffset => limit_offset,
        Ln => ln,
        Log2 => log2,
        Log10 => log10,
        Minute => minute,
        Month => month,
        Now => now,
        Pi => transform_pi,
        PrometheusBuckets => prometheus_buckets,
        Rad => rad,
        Random => rand,
        RandExponential => rand_exp,
        RandNormal => rand_norm,
        RangeAvg => range_avg,
        RangeFirst => range_first,
        RangeLast => range_last,
        RangeLinearRegression => range_linear_regression,
        RangeMAD => range_mad,
        RangeMax => range_max,
        RangeMedian => range_median,
        RangeMin => range_min,
        RangeNormalize => range_normalize,
        RangeQuantile => transform_range_quantile,
        RangeStdDev => range_stddev,
        RangeStdVar => range_stdvar,
        RangeSum => range_sum,
        RangeTrimOutliers => range_trim_outliers,
        RangeTrimSpikes => range_trim_spikes,
        RangeTrimZScore => range_trim_zscore,
        RangeZScore => range_zscore,
        RemoveResets => remove_resets,
        Round => round,
        Ru => transform_ru,
        RunningAvg => running_avg,
        RunningMax => running_max,
        RunningMin => running_min,
        RunningSum => running_sum,
        Scalar => scalar,
        Sgn => sgn,
        Sin => sin,
        Sinh => sinh,
        SmoothExponential => smooth_exponential,
        Sort => sort,
        SortByLabel => sort_by_label,
        SortByLabelDesc => sort_by_label_desc,
        SortByLabelNumeric => sort_alpha_numeric,
        SortByLabelNumericDesc => sort_alpha_numeric_desc,
        SortDesc => sort_desc,
        Sqrt => sqrt,
        Start => transform_start,
        Step => step,
        Tan => tan,
        Tanh => tanh,
        Time => time,
        TimezoneOffset => timezone_offset,
        Union => union,
        Vector => vector,
        Year => year,
    }
}

/// These functions don't change the physical meaning of input time series,
/// so they don't drop metric name
const fn transform_func_keeps_metric_name(func: TransformFunction) -> bool {
    use TransformFunction::*;
    matches!(
        func,
        Ceil | Clamp
            | ClampMax
            | ClampMin
            | Floor
            | Interpolate
            | KeepLastValue
            | KeepNextValue
            | RangeAvg
            | RangeFirst
            | RangeLast
            | RangeLinearRegression
            | RangeMax
            | RangeMedian
            | RangeMin
            | RangeNormalize
            | RangeQuantile
            | RangeStdDev
            | RangeStdVar
            | Round
            | Ru
            | RunningAvg
            | RunningMax
            | RunningMin
            | SmoothExponential
    )
}

pub(crate) fn exec_transform_fn(
    f: TransformFunction,
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let func = get_transform_func(f);
    func(tfa)
}

pub(crate) fn transform_series(
    tfa: &mut TransformFuncArg,
    tf: impl TransformValuesFn,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = tfa.get_param_series(0)?;
    do_transform_values(&mut series, tf, tfa.fe)
}

pub(super) fn do_transform_values(
    arg: &mut Vec<Timeseries>,
    mut tf: impl TransformValuesFn,
    fe: &FunctionExpr,
) -> RuntimeResult<Vec<Timeseries>> {
    let keep_metric_names = match fe.function {
        BuiltinFunction::Transform(tfn) => {
            fe.keep_metric_names || transform_func_keeps_metric_name(tfn)
        }
        _ => fe.keep_metric_names,
    };

    for ts in arg.iter_mut() {
        if !keep_metric_names {
            ts.metric_name.reset_measurement();
        }
        tf(&mut ts.values);
    }
    Ok(std::mem::take(arg))
}
