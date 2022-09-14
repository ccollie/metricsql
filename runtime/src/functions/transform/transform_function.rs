use std::fmt::{Display, Formatter};
use std::str::FromStr;
use phf::phf_map;
use crate::functions::types::{DataType, MAX_ARG_COUNT, Signature, Volatility};
use crate::runtime_error::RuntimeError;

// TODO: tfu, ttf, ru, alias

/// Transform functions calculate transformations over rollup results.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum TransformFunction {
    Abs,
    Absent,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    BitmapAnd,
    BitmapOr,
    BitmapXor,
    BucketsLimit,
    Ceil,
    Clamp,
    ClampMax,
    ClampMin,
    Cos,
    Cosh,
    DayOfMonth,
    DayOfWeek,
    DaysInMonth,
    Deg,
    DropCommonLabels,
    End,
    Exp,
    Floor,
    HistogramAvg,
    HistogramQuantile,
    HistogramQuantiles,
    HistogramShare,
    HistogramStddev,
    HistogramStdvar,
    Hour,
    Interpolate,
    KeepLastValue,
    KeepNextValue,
    LabelCopy,
    LabelDel,
    LabelGraphiteGroup,
    LabelJoin,
    LabelKeep,
    LabelLowercase,
    LabelMap,
    LabelMatch,
    LabelMismatch,
    LabelMove,
    LabelReplace,
    LabelSet,
    LabelTransform,
    LabelUppercase,
    LabelValue,
    LimitOffset,
    Ln,
    Log2,
    Log10,
    Minute,
    Month,
    Now,
    Pi,
    PrometheusBuckets,
    Rad,
    Random,
    RandExponential,
    RandNormal,
    RangeAvg,
    RangeFirst,
    RangeLast,
    RangeMax,
    RangeMin,
    RangeQuantile,
    RangeSum,
    RemoveResets,
    Round,
    RunningAvg,
    RunningMax,
    RunningMin,
    RunningSum,
    Scalar,
    Sgn,
    Sin,
    Sinh,
    SmoothExponential,
    Sort,
    SortByLabel,
    SortByLabelDesc,
    SortByLabelNumeric,
    SortByLabelNumericDesc,
    SortDesc,
    Sqrt,
    Start,
    Step,
    Tan,
    Tanh,
    Time,
    // "timestamp" has been moved to rollup funcs. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415
    TimezoneOffset,
    Union,
    Vector,
    Year
}

impl Display for TransformFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TransformFunction::Abs => "abs",
            TransformFunction::Absent => "absent",
            TransformFunction::Acos => "acos",
            TransformFunction::Acosh => "acosh",
            TransformFunction::Asin => "asin",
            TransformFunction::Asinh => "asinh",
            TransformFunction::Atan => "atan",
            TransformFunction::Atanh => "atanh",
            TransformFunction::BitmapAnd => "bitmap_and",
            TransformFunction::BitmapOr => "bitmap_or",
            TransformFunction::BitmapXor => "bitmap_xor",
            TransformFunction::BucketsLimit => "buckets_limit",
            TransformFunction::Ceil => "ceil",
            TransformFunction::Clamp => "clamp",
            TransformFunction::ClampMax => "clamp_max",
            TransformFunction::ClampMin => "clamp_min",
            TransformFunction::Cos => "cos",
            TransformFunction::Cosh => "cosh",
            TransformFunction::DayOfMonth => "day_of_month",
            TransformFunction::DayOfWeek => "day_of_week",
            TransformFunction::DaysInMonth => "days_in_month",
            TransformFunction::Deg => "deg",
            TransformFunction::DropCommonLabels => "drop_common_labels",
            TransformFunction::End => "end",
            TransformFunction::Exp => "exp",
            TransformFunction::Floor => "floor",
            TransformFunction::HistogramAvg => "histogram_avg",
            TransformFunction::HistogramQuantile => "histogram_quantile",
            TransformFunction::HistogramQuantiles => "histogram_quantiles",
            TransformFunction::HistogramShare => "histogram_share",
            TransformFunction::HistogramStddev => "histogram_stddev",
            TransformFunction::HistogramStdvar => "histogram_stdvar",
            TransformFunction::Hour => "hour",
            TransformFunction::Interpolate => "interpolate",
            TransformFunction::KeepLastValue => "keep_last_value",
            TransformFunction::KeepNextValue => "keep_next_value",
            TransformFunction::LabelCopy => "label_copy",
            TransformFunction::LabelDel => "label_del",
            TransformFunction::LabelGraphiteGroup => "label_graphite_group",
            TransformFunction::LabelJoin => "label_join",
            TransformFunction::LabelKeep => "label_keep",
            TransformFunction::LabelLowercase => "label_lowercase",
            TransformFunction::LabelMap => "label_map",
            TransformFunction::LabelMatch => "label_match",
            TransformFunction::LabelMismatch => "label_mismatch",
            TransformFunction::LabelMove => "label_move",
            TransformFunction::LabelReplace => "label_replace",
            TransformFunction::LabelSet => "label_set",
            TransformFunction::LabelTransform => "label_transform",
            TransformFunction::LabelUppercase => "label_uppercase",
            TransformFunction::LabelValue => "label_value",
            TransformFunction::LimitOffset => "limit_offset",
            TransformFunction::Ln => "ln",
            TransformFunction::Log2 => "log2",
            TransformFunction::Log10 => "log10",
            TransformFunction::Minute => "minute",
            TransformFunction::Month => "month",
            TransformFunction::Now => "now",
            TransformFunction::Pi => "pi",
            TransformFunction::PrometheusBuckets => "prometheus_buckets",
            TransformFunction::Rad => "rad",
            TransformFunction::Random => "rand",
            TransformFunction::RandExponential => "rand_exponential",
            TransformFunction::RandNormal => "rand_normal",
            TransformFunction::RangeAvg => "range_avg",
            TransformFunction::RangeFirst => "range_first",
            TransformFunction::RangeLast => "range_last",
            TransformFunction::RangeMax => "range_max",
            TransformFunction::RangeMin => "range_min",
            TransformFunction::RangeQuantile => "range_quantile",
            TransformFunction::RangeSum => "range_sum",
            TransformFunction::RemoveResets => "remove_resets",
            TransformFunction::Round => "round",
            TransformFunction::RunningAvg => "running_avg",
            TransformFunction::RunningMax => "running_max",
            TransformFunction::RunningMin => "running_min",
            TransformFunction::RunningSum => "running_sum",
            TransformFunction::Scalar => "scalar",
            TransformFunction::Sgn => "sgn",
            TransformFunction::Sin => "sin",
            TransformFunction::Sinh => "sinh",
            TransformFunction::SmoothExponential => "smooth_exponential",
            TransformFunction::Sort => "sort",
            TransformFunction::SortByLabel => "sort_by_label",
            TransformFunction::SortByLabelDesc => "sort_by_label_desc",
            TransformFunction::SortByLabelNumeric => "sort_by_label_numeric",
            TransformFunction::SortByLabelNumericDesc => "sort_by_label_numeric_desc",
            TransformFunction::SortDesc => "sort_desc",
            TransformFunction::Sqrt => "sqrt",
            TransformFunction::Start => "start",
            TransformFunction::Step => "step",
            TransformFunction::Tan => "tan",
            TransformFunction::Tanh => "tanh",
            TransformFunction::Time => "time",
            TransformFunction::TimezoneOffset => "timezone_offset",
            TransformFunction::Union => "union",
            TransformFunction::Vector => "vector",
            TransformFunction::Year => "vector",
        };

        write!(f, "{}", name)
    }
}

static REVERSE_MAP: phf::Map<&'static str, TransformFunction> = phf_map! {
""=> TransformFunction::Union, // empty func is a synonym to union
"abs"=>  TransformFunction::Abs,
"absent"=> TransformFunction::Absent,
"acos"=>  TransformFunction::Acos,
"acosh"=>  TransformFunction::Acosh,
"asin"=>  TransformFunction::Asin,
"asinh" => TransformFunction::Asinh,
"atan" => TransformFunction::Atan,
"atanh" => TransformFunction::Atanh,
"bitmap_and"=> TransformFunction::BitmapAnd,
"bitmap_or"=> TransformFunction::BitmapOr,
"bitmap_xor"=> TransformFunction::BitmapXor,
"buckets_limit"=> TransformFunction::BucketsLimit,
"ceil" => TransformFunction::Ceil,
"clamp" => TransformFunction::Clamp,
"clamp_max" => TransformFunction::ClampMax,
"clamp_min" => TransformFunction::ClampMin,
"cos" => TransformFunction::Cos,
"cosh" => TransformFunction::Cosh,
"day_of_month"=> TransformFunction::DayOfMonth,
"day_of_week"=> TransformFunction::DayOfWeek,
"days_in_month"=> TransformFunction::DaysInMonth,
"deg"=> TransformFunction::Deg,
"drop_common_labels"=> TransformFunction::DropCommonLabels,
"end" => TransformFunction::End,
"exp" => TransformFunction::Exp,
"floor" => TransformFunction::Floor,
"histogram_avg" => TransformFunction::HistogramAvg,
"histogram_quantile" => TransformFunction::HistogramQuantile,
"histogram_quantiles" => TransformFunction::HistogramQuantiles,
"histogram_share" => TransformFunction::HistogramShare,
"histogram_stddev" => TransformFunction::HistogramStddev,
"histogram_stdvar" => TransformFunction::HistogramStdvar,
"hour" => TransformFunction::Hour,
"interpolate" => TransformFunction::Interpolate,
"keep_last_value" => TransformFunction::KeepLastValue,
"keep_next_value" => TransformFunction::KeepNextValue,
"label_copy" => TransformFunction::LabelCopy,
"label_del" =>  TransformFunction::LabelDel,
"label_graphite_group" => TransformFunction::LabelGraphiteGroup,
"label_join" => TransformFunction::LabelJoin,
"label_keep" => TransformFunction::LabelKeep,
"label_lowercase" => TransformFunction::LabelLowercase,
"label_map" => TransformFunction::LabelMap,
"label_match" => TransformFunction::LabelMatch,
"label_mismatch" => TransformFunction::LabelMismatch,
"label_move" => TransformFunction::LabelMove,
"label_replace" => TransformFunction::LabelReplace,
"label_set" => TransformFunction::LabelSet,
"label_transform" => TransformFunction::LabelTransform,
"label_uppercase" => TransformFunction::LabelUppercase,
"label_value" => TransformFunction::LabelValue,
"limit_offset" => TransformFunction::LimitOffset,
"ln" => TransformFunction::Ln,
"log2" => TransformFunction::Log2,
"log10" => TransformFunction::Log10,
"minute" => TransformFunction::Minute,
"month" => TransformFunction::Month,
"now" => TransformFunction::Now,
"pi" => TransformFunction::Pi,
"prometheus_buckets" => TransformFunction::PrometheusBuckets,
"rad" => TransformFunction::Rad,
"rand" => TransformFunction::Random,
"rand_exponential" => TransformFunction::RandExponential,
"rand_normal" => TransformFunction::RandNormal,
"range_avg" => TransformFunction::RunningAvg,
"range_first" => TransformFunction::RangeFirst,
"range_last" => TransformFunction::RangeLast,
"range_max" => TransformFunction::RunningMax,
"range_min" => TransformFunction::RunningMin,
"range_quantile" => TransformFunction::RangeQuantile,
"range_sum" => TransformFunction::RunningSum,
"remove_resets" => TransformFunction::RemoveResets,
"round" => TransformFunction::Round,
"running_avg" => TransformFunction::RunningAvg,
"running_max" => TransformFunction::RunningMax,
"running_min" => TransformFunction::RunningMin,
"running_sum" => TransformFunction::RunningSum,
"scalar" => TransformFunction::Scalar,
"sgn" => TransformFunction::Sgn,
"sin" => TransformFunction::Sin,
"sinh" => TransformFunction::Sinh,
"smooth_exponential" => TransformFunction::SmoothExponential,
"sort" => TransformFunction::Sort,
"sort_by_label" => TransformFunction::SortByLabel,
"sort_by_label_desc" => TransformFunction::SortByLabelDesc,
"sort_by_label_numeric" => TransformFunction::SortByLabelNumeric,
"sort_by_label_numeric_desc" => TransformFunction::SortByLabelNumericDesc,
"sort_desc" => TransformFunction::SortDesc,
"sqrt" => TransformFunction::Sqrt,
"start" => TransformFunction::Start,
"step" => TransformFunction::Step,
"tan" => TransformFunction::Tan,
"tanh" => TransformFunction::Tanh,
"time" => TransformFunction::Time,
// "timestamp" has been moved to rollup funcs. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415
"timezone_offset" => TransformFunction::TimezoneOffset,
"union" => TransformFunction::Union,
"vector" => TransformFunction::Vector,
"year" => TransformFunction::Year
};

impl FromStr for TransformFunction {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match REVERSE_MAP.get(lower.as_str()) {
            Some(op) => Ok(*op),
            None => Err(RuntimeError::AggregateError(
                format!("Invalid rollup function: {}", s)))
        }
    }
}

impl TransformFunction {
    /// These functions don't change physical meaning of input time series,
    /// so they don't drop metric name
    pub fn keep_metric_name(&self) -> bool {
        match self {
            TransformFunction::Ceil |
            TransformFunction::Clamp |
            TransformFunction::ClampMin |
            TransformFunction::ClampMax |
            TransformFunction::Floor |
            TransformFunction::Interpolate |
            TransformFunction::KeepLastValue |
            TransformFunction::KeepNextValue |
            TransformFunction::RangeAvg |
            TransformFunction::RangeFirst |
            TransformFunction::RangeLast |
            TransformFunction::RangeMax |
            TransformFunction::RangeMin |
            TransformFunction::RangeQuantile |
            TransformFunction::Round |
            TransformFunction::RunningAvg |
            TransformFunction::RangeMax |
            TransformFunction::RangeMin |
            TransformFunction::SmoothExponential => true,
            _=> false
        }
    }

    pub fn signature(&self) -> Signature {
        // note: the physical expression must accept the type returned by this function or the execution panics.
        match self {
            TransformFunction::BitmapAnd |
            TransformFunction::BitmapOr |
            TransformFunction::BitmapXor => {
                Signature::exact(vec![DataType::Series, DataType::Int], Volatility::Immutable)
            }
            TransformFunction::BucketsLimit => {
                Signature::exact(vec![DataType::Int, DataType::Series], Volatility::Immutable)
            }
            TransformFunction::Clamp => {
                Signature::exact(vec![DataType::Series, DataType::Int, DataType::Int], Volatility::Volatile)
            }
            TransformFunction::ClampMax |
            TransformFunction::ClampMin => {
                Signature::exact(vec![DataType::Series, DataType::Int], Volatility::Immutable)
            }
            TransformFunction::Start |
            TransformFunction::End => {
                Signature::exact(vec![], Volatility::Stable)
            }
            TransformFunction::DropCommonLabels => {
                Signature::variadic_equal(DataType::Series, 1, Volatility::Stable)
            }
            TransformFunction::HistogramQuantile => {
                Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Stable)
            }
            TransformFunction::HistogramQuantiles => {
                todo!()
            }
            // histogram_share(le, buckets)
            TransformFunction::HistogramShare => {
                Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Stable)
            }
            TransformFunction::LabelCopy |
            TransformFunction::LabelDel |
            TransformFunction::LabelJoin |
            TransformFunction::LabelKeep |
            TransformFunction::LabelLowercase |
            TransformFunction::LabelMap |
            TransformFunction::LabelMatch |
            TransformFunction::LabelMove |
            TransformFunction::LabelSet |
            TransformFunction::LabelLowercase => {
                let mut types = vec![DataType::String; MAX_ARG_COUNT];
                types.insert(0, DataType::Series);
                Signature::exact(types, Volatility::Stable)
            }
            TransformFunction::LabelGraphiteGroup => {
                // label_graphite_group(q, groupNum1, ... groupNumN)
                let mut types = vec![DataType::Int; MAX_ARG_COUNT];
                types.insert(0, DataType::Series);
                Signature::exact(types, Volatility::Stable)
            }
            TransformFunction::LabelReplace => {
                // label_replace(q, "dst_label", "replacement", "src_label", "regex")
                Signature::exact(vec![
                    DataType::Series,
                    DataType::String,
                    DataType::String,
                    DataType::String,
                    DataType::String
                ], Volatility::Stable)
            }
            TransformFunction::LabelTransform => {
                // label_transform(q, "label", "regexp", "replacement")
                Signature::exact(vec![
                    DataType::Series,
                    DataType::String,
                    DataType::String,
                    DataType::String
                ], Volatility::Stable)
            }
            TransformFunction::LabelValue => {
                Signature::exact(vec![DataType::Series, DataType::String], Volatility::Stable)
            }
            TransformFunction::LimitOffset => {
                Signature::exact(vec![DataType::Int, DataType::Int, DataType::Series], Volatility::Stable)
            }
            TransformFunction::Now => {
                Signature::exact(vec![], Volatility::Stable)
            }
            TransformFunction::Pi => {
                Signature::exact(vec![], Volatility::Immutable)
            }
            TransformFunction::Random |
            TransformFunction::RandExponential |
            TransformFunction::RandNormal => {
                Signature::variadic_min(vec![DataType::Float], 0, Volatility::Stable)
            }
            TransformFunction::RangeQuantile => {
                Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Stable)
            }
            TransformFunction::Round => {
                Signature::exact(vec![DataType::Series, DataType::Float], Volatility::Stable)
            }
            TransformFunction::SmoothExponential => {
                Signature::exact(vec![DataType::Series, DataType::Float], Volatility::Stable)
            }
            TransformFunction::SortByLabel |
            TransformFunction::SortByLabelDesc |
            TransformFunction::SortByLabelNumeric |
            TransformFunction::SortByLabelNumericDesc => {
                let mut types = vec![DataType::String; MAX_ARG_COUNT];
                types.insert(0, DataType::Series);
                Signature::exact(types, Volatility::Stable)
            }
            TransformFunction::Step => {
                Signature::exact(vec![], Volatility::Stable)
            }
            TransformFunction::Time => {
                Signature::exact(vec![], Volatility::Stable)
            }
            TransformFunction::TimezoneOffset => {
                Signature::exact(vec![DataType::String], Volatility::Stable)
            }
            TransformFunction::Union => {
                // todo: specify minimum
                Signature::uniform(MAX_ARG_COUNT, DataType::Series, Volatility::Stable)
            }
            TransformFunction::Vector => {
                Signature::exact(vec![DataType::Series], Volatility::Stable)
            }
            _ => {
                // by default we take a single arg containing series
                Signature::exact(vec![DataType::Series], Volatility::Immutable)
            }
        }
    }
}

