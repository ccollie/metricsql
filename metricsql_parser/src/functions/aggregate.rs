//! AggregateFunction module contains enum for available aggregation AggregateFunctions.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use phf::phf_map;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;

use crate::common::ValueType;
use crate::functions::signature::{Signature, Volatility};
use crate::functions::MAX_ARG_COUNT;
use crate::parser::ParseError;

/// Aggregation AggregateFunctions
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, EnumIter, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// calculate the average over dimensions
    Avg,
    /// smallest k elements by sample value
    Bottomk,
    /// count the number of elements in the vector
    Count,
    /// count the number of elements with the same value
    CountValues,
    /// calculate maximum over dimensions
    Max,
    /// calculate minimum over dimensions
    Min,
    /// calculate population standard deviation over dimensions
    StdDev,
    /// calculate population standard variance over dimensions
    StdVar,
    /// largest k elements by sample value
    Topk,
    Group,
    /// calculate φ-quantile (0 ≤ φ ≤ 1) over dimensions
    Quantile,
    Quantiles,
    /// calculate sum over dimensions
    Sum,
    // PromQL extension funcs
    /// any(q) by (group_labels) returns a single series per group_labels out of time series returned by q.
    /// See also group.
    Any,
    BottomkMin,
    BottomkMax,
    BottomkAvg,
    BottomkLast,
    BottomkMedian,
    Distinct,
    GeoMean,
    Histogram,
    Limitk,
    MAD,
    Median,
    Mode,
    OutliersIQR,
    Outliersk,
    OutliersMAD,
    Sum2,
    TopkMin,
    TopkMax,
    TopkAvg,
    TopkLast,
    TopkMedian,
    Share,
    ZScore,
}

impl AggregateFunction {
    pub fn signature(&self) -> Signature {
        aggregate_function_signature(self)
    }

    pub const fn may_sort_results(&self) -> bool {
        use AggregateFunction::*;
        matches!(
            self,
            Topk | Bottomk
                | Outliersk
                | TopkMax
                | TopkMin
                | TopkAvg
                | TopkMedian
                | TopkLast
                | BottomkMax
                | BottomkMin
                | BottomkAvg
                | BottomkMedian
                | BottomkLast
        )
    }

    pub const fn name(&self) -> &'static str {
        use AggregateFunction::*;

        match self {
            Any => "any",
            Avg => "avg",
            Bottomk => "bottomk",
            BottomkAvg => "bottomk_avg",
            BottomkLast => "bottomk_last",
            BottomkMax => "bottomk_max",
            BottomkMedian => "bottomk_median",
            BottomkMin => "bottomk_min",
            Count => "count",
            CountValues => "count_values",
            Distinct => "distinct",
            GeoMean => "geomean",
            Group => "group",
            Histogram => "histogram",
            Limitk => "limitk",
            MAD => "mad",
            Max => "max",
            Median => "median",
            Min => "min",
            Mode => "mode",
            OutliersIQR => "outliers_iqr",
            Outliersk => "outliersk",
            OutliersMAD => "outliers_mad",
            Quantile => "quantile",
            Quantiles => "quantiles",
            Share => "share",
            StdDev => "stddev",
            StdVar => "stdvar",
            Sum => "sum",
            Sum2 => "sum2",
            Topk => "topk",
            TopkMin => "topk_min",
            TopkMax => "topk_max",
            TopkAvg => "topk_avg",
            TopkLast => "topk_last",
            TopkMedian => "topk_median",
            ZScore => "score",
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Display for AggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

static FUNCTION_MAP: phf::Map<&'static str, AggregateFunction> = phf_map! {
    "avg" =>           AggregateFunction::Avg,
    "bottomk" =>       AggregateFunction::Bottomk,
    "bottomk_last" =>  AggregateFunction::BottomkLast,
    "count" =>         AggregateFunction::Count,
    "count_values" =>  AggregateFunction::CountValues,
    "group" =>         AggregateFunction::Group,
    "max" =>           AggregateFunction::Max,
    "min" =>           AggregateFunction::Min,
    "quantile" =>      AggregateFunction::Quantile,
    "quantiles" =>     AggregateFunction::Quantiles,
    "stddev" =>        AggregateFunction::StdDev,
    "stdvar" =>        AggregateFunction::StdVar,
    "sum" =>           AggregateFunction::Sum,
    "topk" =>          AggregateFunction::Topk,

    // PromQL extension funcs
    "any" =>             AggregateFunction::Any,
    "bottomk_min" =>     AggregateFunction::BottomkMin,
    "bottomk_max" =>     AggregateFunction::BottomkMax,
    "bottomk_avg" =>     AggregateFunction::BottomkAvg,
    "bottomk_median" =>  AggregateFunction::BottomkMedian,
    "distinct" =>        AggregateFunction::Distinct,
    "geomean" =>         AggregateFunction::GeoMean,
    "histogram" =>       AggregateFunction::Histogram,
    "limitk" =>          AggregateFunction::Limitk,
    "mad" =>             AggregateFunction::MAD,
    "median" =>          AggregateFunction::Median,
    "mode" =>            AggregateFunction::Mode,
    "outliers_iqr" =>    AggregateFunction::OutliersIQR,
    "outliersk" =>       AggregateFunction::Outliersk,
    "outliers_mad" =>    AggregateFunction::OutliersMAD,
    "share" =>           AggregateFunction::Share,
    "sum2" =>            AggregateFunction::Sum2,
    "topk_min" =>        AggregateFunction::TopkMin,
    "topk_max" =>        AggregateFunction::TopkMax,
    "topk_avg" =>        AggregateFunction::TopkAvg,
    "topk_last" =>       AggregateFunction::TopkLast,
    "topk_median" =>     AggregateFunction::TopkMedian,
    "zscore" =>          AggregateFunction::ZScore,
};

pub fn is_aggr_func(func: &str) -> bool {
    AggregateFunction::from_str(func).is_ok()
}

impl FromStr for AggregateFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        FUNCTION_MAP
            .get(s)
            .or_else(|| {
                let lower = s.to_ascii_lowercase();
                FUNCTION_MAP.get(lower.as_str())
            })
            .ok_or_else(|| ParseError::InvalidFunction(s.to_string()))
            .map(|x| *x)
    }
}

/// the signatures supported by the function `fun`.
pub fn aggregate_function_signature(fun: &AggregateFunction) -> Signature {
    use AggregateFunction::*;
    match fun {
        CountValues => Signature::exact(
            vec![ValueType::String, ValueType::InstantVector],
            Volatility::Stable,
        ),
        Topk | Limitk | Outliersk => Signature::exact(
            vec![ValueType::Scalar, ValueType::InstantVector],
            Volatility::Stable,
        ),
        OutliersMAD => Signature::exact(
            vec![ValueType::Scalar, ValueType::InstantVector],
            Volatility::Stable,
        ),
        TopkMin | TopkMax | TopkAvg | TopkMedian | BottomkMin | BottomkMax | BottomkAvg
        | BottomkLast | BottomkMedian => Signature::exact_with_min_args(
            vec![
                ValueType::Scalar,
                ValueType::InstantVector,
                ValueType::String,
            ],
            2,
            Volatility::Stable,
        ),
        Quantile => Signature::exact(
            vec![ValueType::Scalar, ValueType::InstantVector],
            Volatility::Stable,
        ),
        Quantiles => {
            // todo:
            let mut quantile_types: Vec<ValueType> = vec![ValueType::Scalar; MAX_ARG_COUNT];
            quantile_types.insert(0, ValueType::String);
            quantile_types.push(ValueType::InstantVector);
            Signature::variadic_min(quantile_types, 3, Volatility::Volatile)
        }
        _ => Signature::variadic_min(
            vec![ValueType::InstantVector, ValueType::Scalar],
            1,
            Volatility::Stable,
        ),
    }
}

pub fn get_aggregate_arg_idx_for_optimization(
    func: AggregateFunction,
    arg_count: usize,
) -> Option<usize> {
    use AggregateFunction::*;
    // todo: just examine the signature and return the position containing a vector
    match func {
        Bottomk | BottomkAvg | BottomkMax | BottomkMedian | BottomkLast | BottomkMin | Limitk
        | Outliersk | OutliersMAD | Quantile | Topk | TopkAvg | TopkMax | TopkMedian | TopkLast
        | TopkMin => Some(1),
        CountValues => None,
        Quantiles => Some(arg_count - 1),
        _ => Some(0)
    }
}
