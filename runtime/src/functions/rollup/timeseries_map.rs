use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use metricsql::functions::RollupFunction;

use crate::histogram::Histogram;
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

    pub fn update(&mut self, value: f64) {
        let mut inner = self.inner.write().unwrap();
        inner.update(value);
    }

    pub fn update_from_vec(&self, values: &[f64]) {
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
        let mut ts = inner.get_or_create_timeseries(label_name, label_value);
        f(&mut ts)
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

    pub fn visit_values_mut(&self, f: impl FnMut(&mut Timeseries)) {
        let mut inner = self.inner.write().unwrap();
        inner.series.values_mut().for_each(f)
    }

    pub fn visit_non_zero_buckets<'a, F>(&self, f: F)
    where
        F: Fn(&'a str, u64),
    {
        let inner = self.inner.read().unwrap();
        inner.hist.visit_non_zero_buckets(f)
    }

    pub fn is_valid_function(func: RollupFunction) -> bool {
        use RollupFunction::*;
        matches!(func, HistogramOverTime | QuantilesOverTime)
    }
}
