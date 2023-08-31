use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use metricsql::functions::RollupFunction;

use crate::histogram::{Histogram, NonZeroBucket};
use crate::types::{MetricName, Timeseries};

#[derive(Debug)]
pub(crate) struct TimeseriesMap {
    inner: RwLock<MapInner>,
}

#[derive(Clone, Debug)]
struct MapInner {
    origin: Timeseries,
    hist: Histogram,
    pub series: HashMap<String, Timeseries>,
}

impl MapInner {
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
        origin.timestamps = Arc::clone(shared_timestamps);
        origin.values = vec![f64::NAN; ts_len];
        let m: HashMap<String, Timeseries> = HashMap::new();

        MapInner {
            origin,
            hist: Histogram::new(),
            series: m,
        }
    }

    fn update(&mut self, value: f64) {
        self.hist.update(value);
    }

    fn get_or_create_timeseries(&mut self, label_name: &str, label_value: &str) -> &mut Timeseries {
        let value = label_value.to_string();
        let timestamps = &self.origin.timestamps;
        self.series.entry(value).or_insert_with_key(move |value| {
            let values: Vec<f64> = vec![f64::NAN; timestamps.len()];
            let mut ts = Timeseries::with_shared_timestamps(timestamps, &values);
            ts.metric_name.set_tag(label_name, value);
            ts
        })
    }

    fn reset(&mut self) {
        self.hist.reset();
    }
}

impl TimeseriesMap {
    pub fn new(
        keep_metric_names: bool,
        shared_timestamps: &Arc<Vec<i64>>,
        mn_src: &MetricName,
    ) -> Self {
        let inner = MapInner::new(keep_metric_names, shared_timestamps, mn_src);
        TimeseriesMap {
            inner: RwLock::new(inner),
        }
    }

    pub fn update(&self, values: &[f64]) {
        let mut inner = self.inner.write().unwrap();
        for value in values {
            inner.update(*value);
        }
    }

    pub fn with_timeseries(
        &self,
        label_name: &str,
        label_value: &str,
        f: impl Fn(&mut Timeseries),
    ) {
        let mut inner = self.inner.write().unwrap();
        let ts = inner.get_or_create_timeseries(label_name, label_value);
        f(ts)
    }

    /// Copy all timeseries to dst. The map should not be used after this call
    pub fn append_timeseries_to(&self, dst: &mut Vec<Timeseries>) {
        let mut inner = self.inner.write().unwrap();
        for (_, mut ts) in inner.series.drain() {
            dst.push(std::mem::take(&mut ts))
        }
    }

    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.reset();
    }

    pub fn series_len(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.series.len()
    }

    pub fn visit_non_zero_buckets<'a, F, C>(&self, context: &mut C, f: F)
    where
        F: Fn(&'a str, u64, &mut C),
    {
        let inner = self.inner.read().unwrap();
        inner.hist.visit_non_zero_buckets(context, f)
    }

    pub(crate) fn process_rollup(&self, values: &[f64], rollup_idx: usize) {
        let mut inner = self.inner.write().unwrap();
        inner.reset();
        for value in values {
            inner.update(*value);
        }
        let buckets = inner
            .hist
            .non_zero_buckets()
            .map(|NonZeroBucket { vm_range, count }| (vm_range.to_string(), count))
            .collect::<Vec<_>>();

        for (vm_range, count) in buckets {
            let ts = inner.get_or_create_timeseries("vmrange", &vm_range);
            ts.values[rollup_idx] = count as f64;
        }
    }

    pub(crate) fn set_timeseries_values(
        &self,
        label_name: &str,
        label_values: &[String],
        values: &[f64],
        rollup_idx: usize,
    ) {
        let mut inner = self.inner.write().unwrap();
        for (label_value, value) in label_values.iter().zip(values.iter()) {
            let ts = inner.get_or_create_timeseries(label_name, &label_value);
            ts.values[rollup_idx] = *value;
        }
    }

    pub fn is_valid_function(func: RollupFunction) -> bool {
        use RollupFunction::*;
        matches!(func, HistogramOverTime | QuantilesOverTime)
    }
}
