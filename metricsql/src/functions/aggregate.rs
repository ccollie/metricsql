//! AggregateFunction module contains enum for available aggregation AggregateFunctions.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use phf::phf_map;

use crate::ast::ReturnType;
use crate::functions::data_type::DataType;
use crate::functions::MAX_ARG_COUNT;
use crate::functions::signature::{Signature, Volatility};
use crate::parser::ParseError;
use serde::{Serialize, Deserialize};

/// Aggregation AggregateFunctions
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// calculate sum over dimensions
    Sum,
    /// calculate minimum over dimensions
    Min,
    /// calculate maximum over dimensions
    Max,
    /// calculate the average over dimensions
    Avg,
    /// calculate population standard deviation over dimensions
    StdDev,
    /// calculate population standard variance over dimensions
    StdVar,
    /// count the number of elements in the vector
    Count,
    /// count the number of elements with the same value
    CountValues,
    /// smallest k elements by sample value
    Bottomk,
    /// largest k elements by sample value
    Topk,
    /// calculate φ-quantile (0 ≤ φ ≤ 1) over dimensions
    Quantile,
    Quantiles,
    Group,

    // PromQL extension funcs
    Median,
    MAD,
    Limitk,
    Distinct,
    Sum2,
    GeoMean,
    Histogram,
    TopkMin,
    TopkMax,
    TopkAvg,
    TopkLast,
    TopkMedian,
    BottomkMin,
    BottomkMax,
    BottomkAvg,
    BottomkLast,
    BottomkMedian,
    /// any(q) by (group_labels) returns a single series per group_labels out of time series returned by q.
    /// See also group.
    Any,
    Outliersk,
    OutliersMAD,
    Mode,
    ZScore,
}

impl AggregateFunction {
    pub fn signature(&self) -> Signature {
        aggregate_function_signature(self)
    }

    pub fn sorts_results(&self) -> bool {
        use AggregateFunction::*;
        matches!(self,
        Topk | Bottomk | Outliersk |  TopkMax | TopkMin | TopkAvg |
        TopkMedian | TopkLast | BottomkMax |
        BottomkMin | BottomkAvg | BottomkMedian | BottomkLast
    )
    }

    pub fn name(&self) -> String {
        self.to_string()
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::InstantVector
    }
}

impl Display for AggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use AggregateFunction::*;

        let display = match self {
            Sum => "sum",
            Min => "min",
            Max => "max",
            Avg => "avg",
            StdDev => "stddev",
            StdVar => "stdvar",
            Count => "count",
            CountValues => "count_values",
            Bottomk => "bottomk",
            Topk => "topk",
            Quantile => "quantile",
            Quantiles => "quantiles",
            Group => "group",
            Median => "median",
            MAD => "mad",
            Limitk => "limitk",
            Distinct => "distinct",
            Sum2 => "sum2",
            GeoMean => "geomean",
            Histogram => "histogram",
            TopkMin => "topk_min",
            TopkMax => "topk_max",
            TopkAvg => "topk_avg",
            TopkLast => "topk_last",
            TopkMedian => "topk_median",
            BottomkMin => "bottomk_min",
            BottomkMax => "bottomk_max",
            BottomkAvg => "bottomk_avg",
            BottomkLast => "bottomk_last",
            BottomkMedian => "bottomk_median",
            Any => "any",
            Outliersk => "outliersk",
            OutliersMAD => "outliers_mad",
            Mode => "mode",
            ZScore => "score"
        };

        write!(f, "{}", display)
    }
}

static FUNCTION_MAP: phf::Map<&'static str, AggregateFunction> = phf_map! {
    "sum" =>           AggregateFunction::Sum,
    "min" =>           AggregateFunction::Min,
    "max" =>           AggregateFunction::Max,
    "avg" =>           AggregateFunction::Avg,
    "stddev" =>        AggregateFunction::StdDev,
    "stdvar" =>        AggregateFunction::StdVar,
    "count" =>         AggregateFunction::Count,
    "count_values" =>  AggregateFunction::CountValues,
    "bottomk" =>       AggregateFunction::Bottomk,
    "bottomk_last" =>  AggregateFunction::BottomkLast,
    "topk" =>          AggregateFunction::Topk,
    "quantile" =>      AggregateFunction::Quantile,
    "quantiles" =>     AggregateFunction::Quantiles,
    "group" =>         AggregateFunction::Group,

    // PromQL extension funcs
    "median" =>          AggregateFunction::Median,
    "mad" =>             AggregateFunction::MAD,
    "limitk" =>          AggregateFunction::Limitk,
    "distinct" =>        AggregateFunction::Distinct,
    "sum2" =>            AggregateFunction::Sum2,
    "geomean" =>         AggregateFunction::GeoMean,
    "histogram" =>       AggregateFunction::Histogram,
    "topk_min" =>        AggregateFunction::TopkMin,
    "topk_max" =>        AggregateFunction::TopkMax,
    "topk_avg" =>        AggregateFunction::TopkAvg,
    "topk_last" =>       AggregateFunction::TopkLast,
    "topk_median" =>     AggregateFunction::TopkMedian,
    "bottomk_min" =>     AggregateFunction::BottomkMin,
    "bottomk_max" =>     AggregateFunction::BottomkMax,
    "bottomk_avg" =>     AggregateFunction::BottomkAvg,
    "bottomk_median" =>  AggregateFunction::BottomkMedian,
    "any" =>             AggregateFunction::Any,
    "outliersk" =>       AggregateFunction::Outliersk,
    "outliers_mad" =>    AggregateFunction::OutliersMAD,
    "mode" =>            AggregateFunction::Mode,
    "zscore" =>          AggregateFunction::ZScore,
};

pub fn is_aggr_func(func: &str) -> bool {
    FUNCTION_MAP.contains_key(&func.to_lowercase())
}


impl FromStr for AggregateFunction {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match FUNCTION_MAP.get(lower.as_str()) {
            Some(op) => Ok(*op),
            None => Err(ParseError::InvalidAggregateFunction(
                format!("Invalid aggregation function: {}", s)))
        }
    }
}

/// the signatures supported by the function `fun`.
pub fn aggregate_function_signature(fun: &AggregateFunction) -> Signature {
    use AggregateFunction::*;
    match fun {
        CountValues => {
            Signature::exact(vec![DataType::String, DataType::InstantVector], Volatility::Stable)
        }
        Topk | Limitk | Outliersk => {
            Signature::exact(vec![DataType::Scalar, DataType::InstantVector], Volatility::Stable)
        }
        OutliersMAD => {
            Signature::exact(vec![DataType::Scalar, DataType::InstantVector], Volatility::Stable)
        }
        TopkMin | TopkMax | TopkAvg | TopkMedian |
        BottomkMin | BottomkMax | BottomkAvg | BottomkLast |
        BottomkMedian => {
            Signature::exact(vec![
                DataType::Scalar,
                DataType::InstantVector,
                DataType::String,
            ], Volatility::Stable)
        }
        Quantile => {
            Signature::exact(vec![DataType::Scalar, DataType::InstantVector], Volatility::Stable)
        }
        Quantiles => {
            // todo:
            let mut quantile_types: Vec<DataType> = vec![DataType::Scalar; MAX_ARG_COUNT];
            quantile_types.insert(0, DataType::String);
            quantile_types.push(DataType::InstantVector);
            Signature::variadic_min(quantile_types, 3, Volatility::Volatile)
        }
        _ => {
            Signature::variadic_equal(DataType::InstantVector, 1, Volatility::Stable)
        }
    }
}


pub fn get_aggregate_arg_idx_for_optimization(func: AggregateFunction, arg_count: usize) -> Option<usize> {
    use AggregateFunction::*;
    // todo: just examine the signature and return the position containing a vector
    match func {
        Bottomk | BottomkAvg | BottomkMax | BottomkMedian | BottomkLast | BottomkMin | Limitk |
        Outliersk | OutliersMAD | Quantile | Topk | TopkAvg | TopkMax | TopkMedian | TopkLast | TopkMin => Some(1),
        CountValues => None,
        Quantiles => Some(arg_count - 1),
        _ => Some(0)
    }
}