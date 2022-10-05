use crate::functions::aggregate::aggr_incremental::IncrementalAggrContext;


pub(crate) trait Accumulator {
    fn state(&self) -> &IncrementalAggrContext;
    fn update(values: &[f64]);
    fn merge(src: &IncrementalAggrContext);
    fn finalize();
    // Whether to keep the original MetricName for every time series during aggregation
    fn keep_original() -> bool;
}