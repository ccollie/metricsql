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
use metricsql::prelude::AggregateFunction;

/// all the incremental aggregation functions
/// Using an enum because this needs to be Send
pub enum Handler {
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

impl TryFrom<AggregateFunction> for Handler {
    type Error = String;

    fn try_from(value: AggregateFunction) -> Result<Self, Self::Error> {
        let kind = IncrementalAggrFuncKind::try_from(value)?;
        Ok(Handler::new(kind))
    }
}

impl TryFrom<&str> for Handler {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let kind = IncrementalAggrFuncKind::try_from(value)?;
        Ok(Handler::new(kind))
    }
}

impl Handler {
    pub fn new(func: IncrementalAggrFuncKind) -> Self {
        match func {
            IncrementalAggrFuncKind::Avg => Handler::Avg(IncrementalAggrAvg {}),
            IncrementalAggrFuncKind::Count => Handler::Count(IncrementalAggrCount {}),
            IncrementalAggrFuncKind::Geomean => Handler::Geomean(IncrementalAggrGeomean {}),
            IncrementalAggrFuncKind::Min => Handler::Min(IncrementalAggrMin {}),
            IncrementalAggrFuncKind::Max => Handler::Max(IncrementalAggrMax {}),
            IncrementalAggrFuncKind::Sum => Handler::Sum(IncrementalAggrSum {}),
            IncrementalAggrFuncKind::Sum2 => Handler::Sum2(IncrementalAggrSum2 {}),
            IncrementalAggrFuncKind::Any => Handler::Any(IncrementalAggrAny {}),
            IncrementalAggrFuncKind::Group => Handler::Group(IncrementalAggrGroup {}),
        }
    }
    pub fn update(&self, iac: &mut IncrementalAggrContext, values: &[f64]) {
        match self {
            Handler::Avg(h) => h.update(iac, values),
            Handler::Count(h) => h.update(iac, values),
            Handler::Geomean(h) => h.update(iac, values),
            Handler::Min(h) => h.update(iac, values),
            Handler::Max(h) => h.update(iac, values),
            Handler::Sum(h) => h.update(iac, values),
            Handler::Sum2(h) => h.update(iac, values),
            Handler::Any(h) => h.update(iac, values),
            Handler::Group(h) => h.update(iac, values),
        }
    }
    pub fn merge(&self, dst: &mut IncrementalAggrContext, src: &IncrementalAggrContext) {
        match self {
            Handler::Avg(h) => h.merge(dst, src),
            Handler::Count(h) => h.merge(dst, src),
            Handler::Geomean(h) => h.merge(dst, src),
            Handler::Min(h) => h.merge(dst, src),
            Handler::Max(h) => h.merge(dst, src),
            Handler::Sum(h) => h.merge(dst, src),
            Handler::Sum2(h) => h.merge(dst, src),
            Handler::Any(h) => h.merge(dst, src),
            Handler::Group(h) => h.merge(dst, src),
        }
    }
    pub fn finalize(&self, iac: &mut IncrementalAggrContext) {
        match self {
            Handler::Avg(h) => h.finalize(iac),
            Handler::Count(h) => h.finalize(iac),
            Handler::Geomean(h) => h.finalize(iac),
            Handler::Min(h) => h.finalize(iac),
            Handler::Max(h) => h.finalize(iac),
            Handler::Sum(h) => h.finalize(iac),
            Handler::Sum2(h) => h.finalize(iac),
            Handler::Any(h) => h.finalize(iac),
            Handler::Group(h) => h.finalize(iac),
        }
    }
    // Whether to keep the original MetricName for every time series during aggregation
    pub fn keep_original(&self) -> bool {
        match self {
            Handler::Avg(h) => h.keep_original(),
            Handler::Count(h) => h.keep_original(),
            Handler::Geomean(h) => h.keep_original(),
            Handler::Min(h) => h.keep_original(),
            Handler::Max(h) => h.keep_original(),
            Handler::Sum(h) => h.keep_original(),
            Handler::Sum2(h) => h.keep_original(),
            Handler::Any(h) => h.keep_original(),
            Handler::Group(h) => h.keep_original(),
        }
    }
}
