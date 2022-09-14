use std::fmt::{Display, Formatter};
use std::str::FromStr;
use phf::phf_map;
use crate::functions::types::{DataType, Signature, Volatility};
use crate::runtime_error::RuntimeError;

const MAX_ARG_COUNT: usize = 32;

/// Built-in Rollup Functions
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum RollupFunction {
    AbsentOverTime,
    AggrOverTime,
    AscentOverTime,
    AvgOverTime,
    Changes,
    ChangesPrometheus,
    CountEqOverTime,
    CountGtOverTime,
    CountLeOverTime,
    CountNeOverTime,
    CountOverTime,
    DecreasesOverTime,
    DefaultRollup,
    Delta,
    DeltaPrometheus,
    Deriv,
    DerivFast,
    DescentOverTime,
    DistinctOverTime,
    DurationOverTime,
    FirstOverTime,
    GeomeanOverTime,
    HistogramOverTime,
    HoeffdingBoundLower,
    HoeffdingBoundUpper,
    HoltWinters,
    IDelta,
    IDeriv,
    Increase,
    IncreasePrometheus,
    IncreasePure,
    IncreasesOverTime, 
    Integrate,
    IRate, // + rollupFuncsRemoveCounterResets
    Lag,
    LastOverTime, 
    Lifetime,
    MaxOverTime,
    MinOverTime, 
    ModeOverTime,
    PredictLinear,
    PresentOverTime,
    QuantileOverTime,
    QuantilesOverTime,
    RangeOverTime,
    Rate, // + rollupFuncsRemoveCounterResets
    RateOverSum,
    Resets,
    Rollup,
    RollupCandlestick,
    RollupDelta,
    RollupDeriv,
    RollupIncrease, // + rollupFuncsRemoveCounterResets
    RollupRate,     // + rollupFuncsRemoveCounterResets
    RollupScrapeInterval,
    ScrapeInterval,
    ShareGtOverTime,
    ShareLeOverTime,
    StaleSamplesOverTime,
    StddevOverTime,
    StdvarOverTime,
    SumOverTime,
    Sum2OverTime,
    TFirstOverTime,
    // `timestamp` function must return timestamp for the last datapoint on the current window
    // in order to properly handle offset and timestamps unaligned to the current step.
    // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/415 for details.
    Timestamp,
    TimestampWithName,  // + rollupFuncsKeepMetricName
    TLastChangeOverTime,
    TLastOverTime,
    TMaxOverTime,
    TMinOverTime,
    ZScoreOverTime,
}

impl Display for RollupFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let display = match self {
            RollupFunction::AbsentOverTime => "absent_over_time",
            RollupFunction::AggrOverTime => "aggr_over_time",
            RollupFunction::AscentOverTime => "ascent_over_time",
            RollupFunction::AvgOverTime => "avg_over_time",
            RollupFunction::Changes => "changes",
            RollupFunction::ChangesPrometheus => "changes_prometheus",
            RollupFunction::CountEqOverTime => "count_eq_over_time",
            RollupFunction::CountGtOverTime => "count_gt_over_time",
            RollupFunction::CountLeOverTime => "count_le_over_time",
            RollupFunction::CountNeOverTime => "count_ne_over_time",
            RollupFunction::CountOverTime => "count_over_time",
            RollupFunction::DecreasesOverTime => "decreases_over_time",
            RollupFunction::DefaultRollup => "default_rollup",
            RollupFunction::Delta => "delta",
            RollupFunction::DeltaPrometheus => "delta_prometheus",
            RollupFunction::Deriv => "deriv",
            RollupFunction::DerivFast => "deriv_fast",
            RollupFunction::DescentOverTime => "descent_over_time",
            RollupFunction::DistinctOverTime => "distinct_over_time",
            RollupFunction::DurationOverTime => "duration_over_time",
            RollupFunction::FirstOverTime => "first_over_time",
            RollupFunction::GeomeanOverTime => "geomean_over_time",
            RollupFunction::HistogramOverTime => "histogram_over_time",
            RollupFunction::HoeffdingBoundLower => "hoeffding_bound_lower",
            RollupFunction::HoeffdingBoundUpper => "hoeffding_bound_upper",
            RollupFunction::HoltWinters => "holt_winters",
            RollupFunction::IDelta => "idelta",
            RollupFunction::IDeriv => "ideriv",
            RollupFunction::Increase => "increase",
            RollupFunction::IncreasePrometheus => "increase_prometheus",
            RollupFunction::IncreasePure => "increase_pure",
            RollupFunction::IncreasesOverTime => "increases_over_time",
            RollupFunction::Integrate => "integrate",
            RollupFunction::IRate => "irate",
            RollupFunction::Lag => "lag",
            RollupFunction::LastOverTime => "last_over_time",
            RollupFunction::Lifetime => "lifetime",
            RollupFunction::MaxOverTime => "max_over_time",
            RollupFunction::MinOverTime => "min_over_time",
            RollupFunction::ModeOverTime => "mode_over_time",
            RollupFunction::PredictLinear => "predict_linear",
            RollupFunction::PresentOverTime => "present_over_time",
            RollupFunction::QuantileOverTime => "quantile_over_time",
            RollupFunction::QuantilesOverTime => "quantiles_over_time",
            RollupFunction::RangeOverTime => "range_over_time",
            RollupFunction::Rate => "rate",
            RollupFunction::RateOverSum => "rate_over_sum",
            RollupFunction::Resets => "resets",
            RollupFunction::Rollup => "rollup",
            RollupFunction::RollupCandlestick => "rollup_candlestick",
            RollupFunction::RollupDelta => "rollup_delta",
            RollupFunction::RollupDeriv => "rollup_deriv",
            RollupFunction::RollupIncrease => "rollup_increase",
            RollupFunction::RollupRate => "rollup_rate",
            RollupFunction::RollupScrapeInterval => "rollup_scrape_interval",
            RollupFunction::ScrapeInterval => "scrape_interval",
            RollupFunction::ShareGtOverTime => "share_gt_over_time",
            RollupFunction::ShareLeOverTime => "share_le_over_time",
            RollupFunction::StaleSamplesOverTime => "stale_samples_over_time",
            RollupFunction::StddevOverTime => "stddev_over_time",
            RollupFunction::StdvarOverTime => "stdvar_over_time",
            RollupFunction::SumOverTime => "sum_over_time",
            RollupFunction::Sum2OverTime => "sum2_over_time",
            RollupFunction::TFirstOverTime => "tfirst_over_time",
            RollupFunction::Timestamp => "timestamp",
            RollupFunction::TimestampWithName => "timestamp_with_name",
            RollupFunction::TLastChangeOverTime => "tlast_change_over_time",
            RollupFunction::TLastOverTime => "tlast_over_time",
            RollupFunction::TMaxOverTime => "tmax_over_time",
            RollupFunction::TMinOverTime => "tmin_over_time",
            RollupFunction::ZScoreOverTime => "zscore_over_time",
        };
        write!(f, "{}", display)
    }
}

impl RollupFunction {
    /// the signatures supported by the function `fun`.
    pub fn signature(&self) -> Signature {
        // note: the physical expression must accept the type returned by this function or the execution panics.
        match self {
            RollupFunction::CountEqOverTime
            | RollupFunction::CountLeOverTime
            | RollupFunction::CountNeOverTime
            | RollupFunction::CountGtOverTime
            | RollupFunction::DurationOverTime
            | RollupFunction::PredictLinear
            | RollupFunction::ShareGtOverTime
            | RollupFunction::ShareLeOverTime => {
                Signature::exact(vec![DataType::Series, DataType::Float], Volatility::Immutable)
            },
            RollupFunction::HoeffdingBoundLower | RollupFunction::HoeffdingBoundLower => {
                Signature::exact(vec![DataType::Float, DataType::Series], Volatility::Immutable)
            },
            RollupFunction::HoltWinters => {
                Signature::exact(vec![DataType::Series, DataType::Float, DataType::Float], Volatility::Immutable)
            },
            RollupFunction::AggrOverTime |
            RollupFunction::QuantilesOverTime => {
                let mut quantile_types: Vec<DataType> = vec![DataType::Vector; MAX_ARG_COUNT];
                quantile_types.insert(0, DataType::Series);
                Signature::variadic_min(quantile_types, 3, Volatility::Volatile)
            }
            _ => {
                // default
                Signature::uniform(1, DataType::Series, Volatility::Immutable)
            }
        }      
    }

    /// These functions don't change physical meaning of input time series,
    /// so they don't drop metric name
    pub fn keep_metric_name(&self) -> bool {
        use RollupFunction::*;
        matches!(self,
        AvgOverTime | DefaultRollup | FirstOverTime | GeomeanOverTime | HoeffdingBoundLower |
        HoeffdingBoundUpper | HoltWinters | LastOverTime | MaxOverTime | MinOverTime | ModeOverTime |
        PredictLinear | QuantileOverTime | QuantilesOverTime | Rollup | RollupCandlestick |
        TimestampWithName)
    }

    pub fn should_remove_counter_resets(&self) -> bool {
        use RollupFunction::*;
        matches!(self,
            Increase | IncreasePrometheus | IncreasePure |
            IRate | Rate | RollupIncrease | RollupRate
        )
    }
}


static FUNCTION_MAP: phf::Map<&'static str, RollupFunction> = phf_map! {
    "absent_over_time" => RollupFunction::AbsentOverTime,
    "aggr_over_time" => RollupFunction::AggrOverTime,
    "ascent_over_time" => RollupFunction::AscentOverTime,
    "avg_over_time" => RollupFunction::AvgOverTime,
    "changes" => RollupFunction::Changes,
    "changes_prometheus" => RollupFunction::ChangesPrometheus,
    "count_eq_over_time" => RollupFunction::CountEqOverTime,
    "count_gt_over_time" => RollupFunction::CountGtOverTime,
    "count_le_over_time" => RollupFunction::CountLeOverTime,
    "count_ne_over_time" => RollupFunction::CountNeOverTime,
    "count_over_time" => RollupFunction::CountOverTime,
    "decreases_over_time" => RollupFunction::DecreasesOverTime,
    "default_rollup" => RollupFunction::DefaultRollup,
    "delta" => RollupFunction::Delta,
    "delta_prometheus" => RollupFunction::DeltaPrometheus,
    "deriv" => RollupFunction::Deriv,
    "deriv_fast" => RollupFunction::DerivFast,
    "descent_over_time" => RollupFunction::DescentOverTime,
    "distinct_over_time" => RollupFunction::DistinctOverTime,
    "duration_over_time" => RollupFunction::DurationOverTime,
    "first_over_time" => RollupFunction::FirstOverTime,
    "geomean_over_time" => RollupFunction::GeomeanOverTime,
    "histogram_over_time" => RollupFunction::HistogramOverTime,
    "hoeffding_bound_lower" => RollupFunction::HoeffdingBoundLower,
    "hoeffding_bound_upper" => RollupFunction::HoeffdingBoundUpper,
    "holt_winters" => RollupFunction::HoltWinters,
    "idelta" => RollupFunction::IDelta,
    "ideriv" => RollupFunction::IDeriv,
    "increase" => RollupFunction::Increase,
    "increase_prometheus" => RollupFunction::IncreasePrometheus,
    "increase_pure" => RollupFunction::IncreasePure,
    "increases_over_time" => RollupFunction::IncreasesOverTime,
    "integrate" => RollupFunction::Integrate,
    "irate" => RollupFunction::IRate,
    "lag" => RollupFunction::Lag,
    "last_over_time" => RollupFunction::LastOverTime,
    "lifetime" => RollupFunction::Lifetime,
    "max_over_time" => RollupFunction::MaxOverTime,
    "min_over_time" => RollupFunction::MinOverTime,
    "mode_over_time" => RollupFunction::ModeOverTime,
    "predict_linear" => RollupFunction::PredictLinear,
    "present_over_time" => RollupFunction::PresentOverTime,
    "quantile_over_time" => RollupFunction::QuantileOverTime,
    "quantiles_over_time" => RollupFunction::QuantilesOverTime,
    "range_over_time" => RollupFunction::RangeOverTime,
    "rate" => RollupFunction::Rate,
    "rate_over_sum" => RollupFunction::RateOverSum,
    "resets" => RollupFunction::Resets,
    "rollup" => RollupFunction::Rollup,
    "rollup_candlestick" => RollupFunction::RollupCandlestick,
    "rollup_delta" => RollupFunction::RollupDelta,
    "rollup_deriv" => RollupFunction::RollupDeriv,
    "rollup_increase" => RollupFunction::RollupIncrease,
    "rollup_rate" => RollupFunction::RollupRate,
    "rollup_scrape_interval" => RollupFunction::RollupScrapeInterval,
    "scrape_interval" => RollupFunction::ScrapeInterval,
    "share_gt_over_time" => RollupFunction::ShareGtOverTime,
    "share_le_over_time" => RollupFunction::ShareLeOverTime,
    "stale_samples_over_time" => RollupFunction::StaleSamplesOverTime,
    "stddev_over_time" => RollupFunction::StddevOverTime,
    "stdvar_over_time" => RollupFunction::StdvarOverTime,
    "sum_over_time" => RollupFunction::SumOverTime,
    "sum2_over_time" => RollupFunction::Sum2OverTime,
    "tfirst_over_time" => RollupFunction::TFirstOverTime,
    "timestamp" => RollupFunction::Timestamp,
    "timestamp_with_name" => RollupFunction::TimestampWithName,
    "tlast_change_over_time" => RollupFunction::TLastChangeOverTime,
    "tlast_over_time" => RollupFunction::TLastOverTime,
    "tmax_over_time" => RollupFunction::TMaxOverTime,
    "tmin_over_time" => RollupFunction::TMinOverTime,
    "zscore_over_time" => RollupFunction::ZScoreOverTime,
};


impl FromStr for RollupFunction {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match FUNCTION_MAP.get(lower.as_str()) {
            Some(op) => Ok(*op),
            None => Err(RuntimeError::UnknownFunction(s.to_string()))
        }
    }
}