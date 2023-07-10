use metricsql::prelude::AggregateFunction;

use crate::functions::aggregate::incremental::any::IncrementalAggrAny;
use crate::functions::aggregate::incremental::avg::IncrementalAggrAvg;
use crate::functions::aggregate::incremental::count::IncrementalAggrCount;
use crate::functions::aggregate::incremental::geomean::IncrementalAggrGeomean;
use crate::functions::aggregate::incremental::group::IncrementalAggrGroup;
use crate::functions::aggregate::incremental::max::IncrementalAggrMax;
use crate::functions::aggregate::incremental::min::IncrementalAggrMin;
use crate::functions::aggregate::incremental::sum::IncrementalAggrSum;
use crate::functions::aggregate::incremental::sum2::IncrementalAggrSum2;
use crate::functions::aggregate::{
    IncrementalAggrContext, IncrementalAggrFuncKind, IncrementalAggrHandler,
};

/// all the incremental aggregation functions
/// Using an enum because this needs to be Send
pub enum IncrementalAggregationHandler {
    Avg(IncrementalAggrAvg),
    Count(IncrementalAggrCount),
    Geomean(IncrementalAggrGeomean),
    Min(IncrementalAggrMin),
    Max(IncrementalAggrMax),
    Sum(IncrementalAggrSum),
    Sum2(IncrementalAggrSum2),
    Any(IncrementalAggrAny),
    Group(IncrementalAggrGroup),
}

impl TryFrom<AggregateFunction> for IncrementalAggregationHandler {
    type Error = String;

    fn try_from(value: AggregateFunction) -> Result<Self, Self::Error> {
        let kind = IncrementalAggrFuncKind::try_from(value)?;
        Ok(IncrementalAggregationHandler::new(kind))
    }
}

impl TryFrom<&str> for IncrementalAggregationHandler {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let kind = IncrementalAggrFuncKind::try_from(value)?;
        Ok(IncrementalAggregationHandler::new(kind))
    }
}

impl IncrementalAggregationHandler {
    pub fn handles(func: AggregateFunction) -> bool {
        match func {
            AggregateFunction::Count
            | AggregateFunction::GeoMean
            | AggregateFunction::Min
            | AggregateFunction::Max
            | AggregateFunction::Avg
            | AggregateFunction::Sum
            | AggregateFunction::Sum2
            | AggregateFunction::Any
            | AggregateFunction::Group => true,
            _ => false,
        }
    }

    pub fn new(func: IncrementalAggrFuncKind) -> Self {
        use IncrementalAggregationHandler::*;
        match func {
            IncrementalAggrFuncKind::Avg => Avg(IncrementalAggrAvg {}),
            IncrementalAggrFuncKind::Count => Count(IncrementalAggrCount {}),
            IncrementalAggrFuncKind::Geomean => Geomean(IncrementalAggrGeomean {}),
            IncrementalAggrFuncKind::Min => Min(IncrementalAggrMin {}),
            IncrementalAggrFuncKind::Max => Max(IncrementalAggrMax {}),
            IncrementalAggrFuncKind::Sum => Sum(IncrementalAggrSum {}),
            IncrementalAggrFuncKind::Sum2 => Sum2(IncrementalAggrSum2 {}),
            IncrementalAggrFuncKind::Any => Any(IncrementalAggrAny {}),
            IncrementalAggrFuncKind::Group => Group(IncrementalAggrGroup {}),
        }
    }
    pub fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        use IncrementalAggregationHandler::*;
        match self {
            Avg(h) => h.update(iac, values),
            Count(h) => h.update(iac, values),
            Geomean(h) => h.update(iac, values),
            Min(h) => h.update(iac, values),
            Max(h) => h.update(iac, values),
            Sum(h) => h.update(iac, values),
            Sum2(h) => h.update(iac, values),
            Any(h) => h.update(iac, values),
            Group(h) => h.update(iac, values),
        }
    }
    pub fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        use IncrementalAggregationHandler::*;
        match self {
            Avg(h) => h.merge(dst, src),
            Count(h) => h.merge(dst, src),
            Geomean(h) => h.merge(dst, src),
            Min(h) => h.merge(dst, src),
            Max(h) => h.merge(dst, src),
            Sum(h) => h.merge(dst, src),
            Sum2(h) => h.merge(dst, src),
            Any(h) => h.merge(dst, src),
            Group(h) => h.merge(dst, src),
        }
    }
    pub fn finalize(&self, iac: &mut IncrementalAggrContext) {
        use IncrementalAggregationHandler::*;
        match self {
            Avg(h) => h.finalize(iac),
            Count(h) => h.finalize(iac),
            Geomean(h) => h.finalize(iac),
            Min(h) => h.finalize(iac),
            Max(h) => h.finalize(iac),
            Sum(h) => h.finalize(iac),
            Sum2(h) => h.finalize(iac),
            Any(h) => h.finalize(iac),
            Group(h) => h.finalize(iac),
        }
    }
    // Whether to keep the original MetricName for every time series during aggregation
    pub fn keep_original(&self) -> bool {
        use IncrementalAggregationHandler::*;
        match self {
            Avg(h) => h.keep_original(),
            Count(h) => h.keep_original(),
            Geomean(h) => h.keep_original(),
            Min(h) => h.keep_original(),
            Max(h) => h.keep_original(),
            Sum(h) => h.keep_original(),
            Sum2(h) => h.keep_original(),
            Any(h) => h.keep_original(),
            Group(h) => h.keep_original(),
        }
    }
}
