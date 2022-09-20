
//! AggregateFunction module contains enum for available aggregation AggregateFunctions.

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use phf::phf_map;

use crate::ast::ReturnValue;
use crate::functions::data_type::DataType;
use crate::functions::MAX_ARG_COUNT;
use crate::functions::signature::{Signature, Volatility};
use crate::parser::ParseError;

/// Aggregation AggregateFunctions
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
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
  ZScore
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

  pub fn return_type(&self) -> ReturnValue {
    ReturnValue::InstantVector
  }
}

impl Display for AggregateFunction {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    let display = match self {
      AggregateFunction::Sum => "sum",
      AggregateFunction::Min => "min",
      AggregateFunction::Max => "max",
      AggregateFunction::Avg => "avg",
      AggregateFunction::StdDev => "stddev",
      AggregateFunction::StdVar => "stdvar",
      AggregateFunction::Count => "count",
      AggregateFunction::CountValues => "count_values",
      AggregateFunction::Bottomk => "bottomk",
      AggregateFunction::Topk => "topk",
      AggregateFunction::Quantile => "quantile",
      AggregateFunction::Quantiles => "quantiles",
      AggregateFunction::Group => "group",
      AggregateFunction::Median => "median",
      AggregateFunction::MAD => "mad",
      AggregateFunction::Limitk => "limitk",
      AggregateFunction::Distinct => "distinct",
      AggregateFunction::Sum2 => "sum2",
      AggregateFunction::GeoMean => "geomean",
      AggregateFunction::Histogram => "histogram",
      AggregateFunction::TopkMin => "topk_min",
      AggregateFunction::TopkMax => "topk_max",
      AggregateFunction::TopkAvg => "topk_avg",
      AggregateFunction::TopkLast => "topk_last",
      AggregateFunction::TopkMedian => "topk_median",
      AggregateFunction::BottomkMin => "bottomk_min",
      AggregateFunction::BottomkMax => "bottomk_max",
      AggregateFunction::BottomkAvg => "bottomk_avg",
      AggregateFunction::BottomkLast => "bottomk_last",
      AggregateFunction::BottomkMedian => "bottomk_median",
      AggregateFunction::Any => "any",
      AggregateFunction::Outliersk => "outliersk",
      AggregateFunction::OutliersMAD => "outliers_mad",
      AggregateFunction::Mode => "mode",
      AggregateFunction::ZScore => "score"
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
  match fun {
    AggregateFunction::Bottomk |
    AggregateFunction::Sum |
    AggregateFunction::Min |
    AggregateFunction::Max |
    AggregateFunction::Any |
    AggregateFunction::Avg |
    AggregateFunction::Distinct |
    AggregateFunction::GeoMean |
    AggregateFunction::Group |
    AggregateFunction::Histogram |
    AggregateFunction::MAD |
    AggregateFunction::Median |
    AggregateFunction::Mode |
    AggregateFunction::StdDev |
    AggregateFunction::StdVar |
    AggregateFunction::Sum2 |
    AggregateFunction::ZScore |
    AggregateFunction::Count => {
      Signature::exact(vec![DataType::Series], Volatility::Stable)
    }
    AggregateFunction::CountValues => {
      Signature::exact(vec![DataType::String, DataType::Series], Volatility::Stable)
    }
    AggregateFunction::Topk |
    AggregateFunction::Limitk |
    AggregateFunction::Outliersk => {
      Signature::exact(vec![DataType::Int, DataType::Series], Volatility::Stable)
    }
    AggregateFunction::OutliersMAD => {
      Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Stable)
    }
    AggregateFunction::TopkMin |
    AggregateFunction::TopkMax |
    AggregateFunction::TopkAvg |
    AggregateFunction::TopkMedian |
    AggregateFunction::BottomkMin |
    AggregateFunction::BottomkMax |
    AggregateFunction::BottomkAvg |
    AggregateFunction::BottomkLast |
    AggregateFunction::BottomkMedian => {
      Signature::variadic_min(vec![
        DataType::Int,
        DataType::Series,
        DataType::String
      ], 2, Volatility::Stable)
    }
    AggregateFunction::Quantile => {
      Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Stable)
    }
    AggregateFunction::Quantiles => {
      // todo:
      let mut quantile_types: Vec<DataType> = vec![DataType::Float; MAX_ARG_COUNT];
      quantile_types.insert(0, DataType::String);
      quantile_types.push(DataType::Series);
      Signature::variadic_min(quantile_types, 3, Volatility::Volatile)
    }
    _ => {
      Signature::exact(vec![DataType::Series], Volatility::Stable)
    }
  }
}


fn aggr_function_can_sort_results(func: AggregateFunction) -> bool {
  use AggregateFunction::*;
  matches!(func,
        Topk | Bottomk | Outliersk |  TopkMax | TopkMin | TopkAvg |
        TopkMedian | TopkLast | BottomkMax |
        BottomkMin | BottomkAvg | BottomkMedian | BottomkLast
    )
}

pub fn get_aggregate_arg_idx_for_optimization(func: AggregateFunction, arg_count: usize) -> Option<usize> {
  use AggregateFunction::*;
  match func {
    Bottomk | BottomkAvg | BottomkMax | BottomkMedian | BottomkLast | BottomkMin | Limitk |
    Outliersk | OutliersMAD | Quantile | Topk | TopkAvg | TopkMax | TopkMedian | TopkLast | TopkMin => Some(1),
    CountValues => None,
    Quantiles => Some(arg_count - 1),
    _ => Some(0)
  }
}