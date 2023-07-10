use std::collections::hash_map::ValuesMut;
use std::collections::HashMap;
use std::sync::Arc;

use metricsql::functions::RollupFunction;

use crate::histogram::{Histogram, NonZeroBuckets};
use crate::types::{MetricName, Timeseries};

#[derive(Clone, Debug)]
pub(crate) struct TimeseriesMap {
    origin: Timeseries,
    hist: Histogram,
    pub series: HashMap<String, Timeseries>,
}

impl TimeseriesMap {
    pub fn new(
        keep_metric_names: bool,
        shared_timestamps: &Arc<Vec<i64>>,
        mn_src: &MetricName,
    ) -> Self {
        let ts_len = shared_timestamps.len();

        let mut origin: Timeseries = Timeseries::default();
        origin.metric_name.copy_from(mn_src);
        if !keep_metric_names {
            origin.metric_name.reset_metric_group()
        }
        origin.timestamps = Arc::clone(&shared_timestamps);
        origin.values = vec![f64::NAN; ts_len];
        let m: HashMap<String, Timeseries> = HashMap::new();

        TimeseriesMap {
            origin,
            hist: Histogram::new(),
            series: m,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.hist.update(value);
    }

    pub fn get_or_create_timeseries(
        &mut self,
        label_name: &str,
        label_value: &str,
    ) -> &mut Timeseries {
        let value = label_value.to_string();
        let timestamps = &self.origin.timestamps;
        self.series.entry(value).or_insert_with_key(move |value| {
            let values: Vec<f64> = Vec::with_capacity(1);
            let mut ts = Timeseries::with_shared_timestamps(&timestamps, &values);
            ts.metric_name.set_tag(label_name, value);
            ts
        })
    }

    /// Copy all timeseries to dst. The map should not be used after this call
    pub fn append_timeseries_to(&mut self, dst: &mut Vec<Timeseries>) {
        for (_, mut ts) in self.series.drain() {
            dst.push(std::mem::take(&mut ts))
        }
    }

    pub fn reset(&mut self) {
        self.hist.reset();
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, String, Timeseries> {
        self.series.values_mut()
    }

    pub fn non_zero_buckets(&mut self) -> NonZeroBuckets {
        self.hist.non_zero_buckets()
    }

    pub fn is_valid_function(func: &RollupFunction) -> bool {
        use RollupFunction::*;
        matches!(*func, HistogramOverTime | QuantilesOverTime)
    }
}
