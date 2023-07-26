use std::fmt::{Display, Formatter};
use std::str::FromStr;

use phf::phf_map;
use serde::{Deserialize, Serialize};

use crate::common::ValueType;
use crate::functions::signature::{Signature, Volatility};
use crate::functions::MAX_ARG_COUNT;
use crate::parser::ParseError;

// TODO: ttf

/// Transform functions calculate transformations over rollup results.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Serialize, Deserialize)]
pub enum TransformFunction {
    Abs,
    Absent,
    Acos,
    Acosh,
    Alias,
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
    RangeLinearRegression,
    RangeMax,
    RangeMedian,
    RangeMin,
    RangeNormalize,
    RangeQuantile,
    RangeStdDev,
    RangeStdVar,
    RangeSum,
    RangeTrimOutliers,
    RangeTrimSpikes,
    RangeTrimZscore,
    RangeZscore,
    RemoveResets,
    Round,
    Ru,
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
    Year,
}

impl Display for TransformFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

static REVERSE_MAP: phf::Map<&'static str, TransformFunction> = phf_map! {
""=> TransformFunction::Union, // empty func is a synonym to union
"abs"=>  TransformFunction::Abs,
"absent"=> TransformFunction::Absent,
"acos"=>  TransformFunction::Acos,
"acosh"=>  TransformFunction::Acosh,
"alias"=>  TransformFunction::Alias,
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
"range_avg" => TransformFunction::RangeAvg,
"range_first" => TransformFunction::RangeFirst,
"range_linear_regression" => TransformFunction::RangeLinearRegression,
"range_last" => TransformFunction::RangeLast,
"range_max" => TransformFunction::RangeMax,
"range_median" => TransformFunction::RangeMedian,
"range_min" => TransformFunction::RangeMin,
"range_normalize" => TransformFunction::RangeNormalize,
"range_quantile" => TransformFunction::RangeQuantile,
"range_stddev" => TransformFunction::RangeStdDev,
"range_stdvar" => TransformFunction::RangeStdVar,
"range_sum" => TransformFunction::RangeSum,
"range_trim_outliers" => TransformFunction::RangeTrimOutliers,
"range_trim_spikes" => TransformFunction::RangeTrimSpikes,
"range_trim_zscore" => TransformFunction::RangeTrimZscore,
"range_zscore" => TransformFunction::RangeZscore,
"remove_resets" => TransformFunction::RemoveResets,
"round" => TransformFunction::Round,
"ru" => TransformFunction::Ru,
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
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match REVERSE_MAP.get(lower.as_str()) {
            Some(op) => Ok(*op),
            None => Err(ParseError::InvalidFunction(s.to_string())),
        }
    }
}

impl TransformFunction {
    pub fn name(&self) -> &'static str {
        use TransformFunction::*;
        match self {
            Abs => "abs",
            Absent => "absent",
            Acos => "acos",
            Acosh => "acosh",
            Alias => "alias",
            Asin => "asin",
            Asinh => "asinh",
            Atan => "atan",
            Atanh => "atanh",
            BitmapAnd => "bitmap_and",
            BitmapOr => "bitmap_or",
            BitmapXor => "bitmap_xor",
            BucketsLimit => "buckets_limit",
            Ceil => "ceil",
            Clamp => "clamp",
            ClampMax => "clamp_max",
            ClampMin => "clamp_min",
            Cos => "cos",
            Cosh => "cosh",
            DayOfMonth => "day_of_month",
            DayOfWeek => "day_of_week",
            DaysInMonth => "days_in_month",
            Deg => "deg",
            DropCommonLabels => "drop_common_labels",
            End => "end",
            Exp => "exp",
            Floor => "floor",
            HistogramAvg => "histogram_avg",
            HistogramQuantile => "histogram_quantile",
            HistogramQuantiles => "histogram_quantiles",
            HistogramShare => "histogram_share",
            HistogramStddev => "histogram_stddev",
            HistogramStdvar => "histogram_stdvar",
            Hour => "hour",
            Interpolate => "interpolate",
            KeepLastValue => "keep_last_value",
            KeepNextValue => "keep_next_value",
            LabelCopy => "label_copy",
            LabelDel => "label_del",
            LabelGraphiteGroup => "label_graphite_group",
            LabelJoin => "label_join",
            LabelKeep => "label_keep",
            LabelLowercase => "label_lowercase",
            LabelMap => "label_map",
            LabelMatch => "label_match",
            LabelMismatch => "label_mismatch",
            LabelMove => "label_move",
            LabelReplace => "label_replace",
            LabelSet => "label_set",
            LabelTransform => "label_transform",
            LabelUppercase => "label_uppercase",
            LabelValue => "label_value",
            LimitOffset => "limit_offset",
            Ln => "ln",
            Log2 => "log2",
            Log10 => "log10",
            Minute => "minute",
            Month => "month",
            Now => "now",
            Pi => "pi",
            PrometheusBuckets => "prometheus_buckets",
            Rad => "rad",
            Random => "rand",
            RandExponential => "rand_exponential",
            RandNormal => "rand_normal",
            RangeAvg => "range_avg",
            RangeFirst => "range_first",
            RangeLast => "range_last",
            RangeLinearRegression => "range_linear_regression",
            RangeMax => "range_max",
            RangeMedian => "range_median",
            RangeMin => "range_min",
            RangeNormalize => "range_normalize",
            RangeQuantile => "range_quantile",
            RangeStdDev => "range_stddev",
            RangeStdVar => "range_stdvar",
            RangeSum => "range_sum",
            RangeTrimSpikes => "range_trim_spikes",
            RangeTrimOutliers => "range_trim_outliers",
            RangeTrimZscore => "range_trim_zscore",
            RangeZscore => "range_zscore",
            RemoveResets => "remove_resets",
            Round => "round",
            Ru => "ru",
            RunningAvg => "running_avg",
            RunningMax => "running_max",
            RunningMin => "running_min",
            RunningSum => "running_sum",
            Scalar => "scalar",
            Sgn => "sgn",
            Sin => "sin",
            Sinh => "sinh",
            SmoothExponential => "smooth_exponential",
            Sort => "sort",
            SortByLabel => "sort_by_label",
            SortByLabelDesc => "sort_by_label_desc",
            SortByLabelNumeric => "sort_by_label_numeric",
            SortByLabelNumericDesc => "sort_by_label_numeric_desc",
            SortDesc => "sort_desc",
            Sqrt => "sqrt",
            Start => "start",
            Step => "step",
            Tan => "tan",
            Tanh => "tanh",
            Time => "time",
            TimezoneOffset => "timezone_offset",
            Union => "union",
            Vector => "vector",
            Year => "year",
        }
    }

    /// These functions don't change physical meaning of input time series,
    /// so they don't drop metric name
    pub const fn keep_metric_name(&self) -> bool {
        use TransformFunction::*;
        matches!(
            self,
            Ceil | Clamp
                | ClampMin
                | ClampMax
                | Floor
                | Interpolate
                | KeepLastValue
                | KeepNextValue
                | RangeAvg
                | RangeFirst
                | RangeLast
                | RangeMax
                | RangeMedian
                | RangeMin
                | RangeNormalize
                | RangeQuantile
                | Round
                | Ru
                | RunningAvg
                | SmoothExponential
        )
    }

    pub const fn may_sort_results(&self) -> bool {
        use TransformFunction::*;
        matches!(
            &self,
            Sort | SortDesc
                | SortByLabel
                | SortByLabelDesc
                | SortByLabelNumeric
                | SortByLabelNumericDesc
        )
    }

    pub const fn manipulates_labels(&self) -> bool {
        use TransformFunction::*;
        matches!(
            self,
            Alias
                | DropCommonLabels
                | LabelCopy
                | LabelDel
                | LabelGraphiteGroup
                | LabelJoin
                | LabelKeep
                | LabelLowercase
                | LabelMap
                | LabelMatch
                | LabelMismatch
                | LabelMove
                | LabelReplace
                | LabelSet
                | LabelTransform
                | LabelUppercase
                | LabelValue
        )
    }

    pub fn signature(&self) -> Signature {
        use TransformFunction::*;

        // note: the expression must accept the type returned by this function or the execution panics.
        match self {
            Alias => Signature::exact(
                vec![ValueType::InstantVector, ValueType::String],
                Volatility::Stable,
            ),
            BitmapAnd | BitmapOr | BitmapXor => Signature::exact(
                vec![ValueType::InstantVector, ValueType::Scalar],
                Volatility::Immutable,
            ),
            BucketsLimit => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector],
                Volatility::Immutable,
            ),
            Clamp => Signature::exact(
                vec![
                    ValueType::InstantVector,
                    ValueType::Scalar,
                    ValueType::Scalar,
                ],
                Volatility::Volatile,
            ),
            ClampMax | ClampMin => Signature::exact(
                vec![ValueType::InstantVector, ValueType::Scalar],
                Volatility::Immutable,
            ),
            Start | End => Signature::exact(vec![], Volatility::Stable),
            DropCommonLabels => {
                Signature::variadic_equal(ValueType::InstantVector, 1, Volatility::Stable)
            }
            HistogramQuantile => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector],
                Volatility::Stable,
            ),
            HistogramQuantiles => {
                // histogram_quantiles("phiLabel", phi1, ..., phiN, buckets)
                // todo: need a better way to handle variadic args with specific types
                Signature::variadic_any(3, Volatility::Stable)
            }
            // histogram_share(le, buckets)
            HistogramShare => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector],
                Volatility::Stable,
            ),
            LabelCopy | LabelMove | LabelSet => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 3, Volatility::Stable)
            }
            LabelDel | LabelKeep | LabelLowercase | LabelUppercase => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 2, Volatility::Stable)
            }
            LabelJoin => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 4, Volatility::Stable)
            }
            LabelMap => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 4, Volatility::Stable)
            }
            LabelMatch | LabelMismatch => {
                let types = vec![
                    ValueType::InstantVector,
                    ValueType::String,
                    ValueType::String,
                ];
                Signature::exact_with_min_args(types, 3, Volatility::Stable)
            }
            LabelGraphiteGroup => {
                // label_graphite_group(q, groupNum1, ... groupNumN)
                let mut types = vec![ValueType::Scalar; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact(types, Volatility::Stable)
            }
            LabelReplace => {
                // label_replace(q, "dst_label", "replacement", "src_label", "regex")
                Signature::exact(
                    vec![
                        ValueType::InstantVector,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                    ],
                    Volatility::Stable,
                )
            }
            LabelTransform => {
                // label_transform(q, "label", "regexp", "replacement")
                Signature::exact(
                    vec![
                        ValueType::InstantVector,
                        ValueType::String,
                        ValueType::String,
                        ValueType::String,
                    ],
                    Volatility::Stable,
                )
            }
            LabelValue => Signature::exact(
                vec![ValueType::InstantVector, ValueType::String],
                Volatility::Stable,
            ),
            LimitOffset => Signature::exact(
                vec![
                    ValueType::Scalar,
                    ValueType::Scalar,
                    ValueType::InstantVector,
                ],
                Volatility::Stable,
            ),
            Now => Signature::exact(vec![], Volatility::Stable),
            Pi => Signature::exact(vec![], Volatility::Immutable),
            Random | RandExponential | RandNormal => {
                Signature::variadic_min(vec![ValueType::Scalar], 0, Volatility::Volatile)
            }
            RangeNormalize => {
                Signature::variadic_min(vec![ValueType::InstantVector], 1, Volatility::Stable)
            }
            RangeTrimOutliers | RangeTrimSpikes | RangeTrimZscore => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector],
                Volatility::Stable,
            ),
            RangeQuantile => Signature::exact(
                vec![ValueType::Scalar, ValueType::InstantVector],
                Volatility::Stable,
            ),
            Round => Signature::exact_with_min_args(
                vec![ValueType::InstantVector, ValueType::Scalar],
                1,
                Volatility::Stable,
            ),
            Ru => Signature::exact(
                vec![ValueType::RangeVector, ValueType::RangeVector],
                Volatility::Stable,
            ),
            Scalar => Signature::any(1, Volatility::Stable),
            SmoothExponential => Signature::exact(
                vec![ValueType::InstantVector, ValueType::Scalar],
                Volatility::Stable,
            ),
            SortByLabel | SortByLabelDesc | SortByLabelNumeric | SortByLabelNumericDesc => {
                let mut types = vec![ValueType::String; MAX_ARG_COUNT];
                types.insert(0, ValueType::InstantVector);
                Signature::exact_with_min_args(types, 2, Volatility::Stable)
            }
            Step => Signature::exact(vec![], Volatility::Stable),
            Time => Signature::exact(vec![], Volatility::Stable),
            TimezoneOffset => Signature::exact(vec![ValueType::String], Volatility::Stable),
            Union => {
                // todo: specify minimum
                Signature::uniform(MAX_ARG_COUNT, ValueType::InstantVector, Volatility::Stable)
            }
            Vector => Signature::exact(vec![ValueType::InstantVector], Volatility::Stable),
            // DateTime functions
            DayOfMonth | DayOfWeek | DaysInMonth | Hour | Minute | Month | Year => {
                Signature::exact_with_min_args(
                    vec![ValueType::InstantVector],
                    0,
                    Volatility::Immutable,
                )
            }
            _ => {
                // by default we take a single arg containing series
                Signature::exact(vec![ValueType::InstantVector], Volatility::Immutable)
            }
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

pub const fn get_transform_arg_idx_for_optimization(
    func: TransformFunction,
    arg_count: usize,
) -> Option<usize> {
    if func.manipulates_labels() {
        return None;
    }

    use TransformFunction::*;
    match func {
        Absent | Scalar | Union | Vector => None,
        End | Now | Pi | Start | Step | Time => None, // todo Ru
        LimitOffset => Some(2),
        BucketsLimit | HistogramQuantile | HistogramShare | RangeQuantile | RangeTrimSpikes => {
            Some(1)
        }
        HistogramQuantiles => Some(arg_count - 1),
        _ => Some(0),
    }
}
