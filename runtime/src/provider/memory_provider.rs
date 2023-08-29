use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use itertools::Itertools;
use regex::Regex;

use metricsql::prelude::{LabelFilter, LabelFilterOp, Matchers};

use crate::signature::Signature;
use crate::{
    Deadline, MetricDataProvider, MetricName, QueryResult, QueryResults, RuntimeResult, SearchQuery,
};

#[derive(Debug, Clone)]
pub struct Point {
    t: i64,
    v: f64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub metric: MetricName,
    pub timestamp: i64,
    pub value: f64,
}

/// In-memory implementation of MetricDataProvider primarily for testing
#[derive(Default, Debug)]
pub struct MemoryMetricProvider {
    inner: RwLock<Storage>,
}

#[derive(Default, Debug, Clone)]
struct Storage {
    labels_hash: BTreeMap<Signature, Arc<MetricName>>,
    sample_values: BTreeMap<Signature, Vec<Point>>,
    need_sort: BTreeSet<Signature>,
    pending_values: BTreeMap<Signature, Vec<Sample>>,
}

impl Storage {
    fn append(&mut self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let h = labels.signature();
        match self.labels_hash.entry(h) {
            Entry::Vacant(e) => {
                let metric = Arc::new(labels);
                e.insert(metric);
                self.sample_values.insert(h, vec![Point { t, v }]);
            }
            Entry::Occupied(_) => {
                let values = self.sample_values.entry(h).or_default();
                values.push(Point { t, v });
            }
        }
        Ok(())
    }

    pub fn search(
        &self,
        start: i64,
        end: i64,
        filters: &Vec<Matchers>,
    ) -> RuntimeResult<QueryResults> {
        let mut results: Vec<QueryResult> = vec![];
        for matchers in filters {
            for (k, labels) in &self.labels_hash {
                let matched = matchers.iter().any(|m| matches_filter(labels, m));
                if matched {
                    if let Some(res) = self.get_range(*k, start, end) {
                        results.push(res)
                    }
                }
            }
        }

        Ok(QueryResults::new(results))
    }

    fn get_range(&self, metric_id: Signature, start: i64, end: i64) -> Option<QueryResult> {
        if let Some(values) = self.sample_values.get(&metric_id) {
            if let Some(start) = find_first_index(&values, start) {
                let points = &values[start..]
                    .iter()
                    .filter(|p| p.t <= end)
                    .sorted_by(|a, b| a.t.cmp(&b.t))
                    .collect::<Vec<_>>();

                let mut timestamps = Vec::with_capacity(points.len());
                let mut values = Vec::with_capacity(points.len());

                for point in points {
                    timestamps.push(point.t);
                    values.push(point.v);
                }

                let metric_name = if let Some(mn) = self.labels_hash.get(&metric_id) {
                    let copy: MetricName = mn.deref().clone();
                    copy
                } else {
                    MetricName::default()
                };

                QueryResult {
                    metric: metric_name,
                    values,
                    timestamps,
                    rows_processed: 0,
                    worker_id: 0,
                };
            }
        }
        None
    }

    fn clear(&mut self) {
        self.need_sort.clear();
        self.labels_hash.clear();
        self.sample_values.clear();
    }

    fn commit(&mut self) -> RuntimeResult<()> {
        let mut all_samples = vec![];
        for samples in self.pending_values.values_mut() {
            all_samples.append(samples);
        }
        for sample in all_samples {
            self.append(sample.metric, sample.timestamp, sample.value)?;
        }
        self.pending_values.clear();
        Ok(())
    }

    fn rollback(&mut self) {
        self.pending_values.clear();
    }

    fn sort_if_needed(&mut self, id: Signature) {
        if self.need_sort.contains(&id) {
            self.need_sort.remove(&id);
            if let Some(points) = self.sample_values.get_mut(&id) {
                points.sort_by(|a, b| a.t.cmp(&b.t))
            }
        }
    }
}

impl MemoryMetricProvider {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Storage {
                labels_hash: Default::default(),
                sample_values: Default::default(),
                need_sort: Default::default(),
                pending_values: Default::default(),
            }),
        }
    }

    pub fn append(&mut self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(labels, t, v)
    }

    fn commit(&mut self) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.commit()
    }

    fn rollback(&mut self) {
        let mut inner = self.inner.write().unwrap();
        inner.rollback();
    }

    pub fn add_sample(&mut self, sample: Sample) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(sample.metric, sample.timestamp, sample.value)
    }

    pub fn clear(&mut self) {
        let mut inner = self.inner.write().unwrap();
        inner.clear();
    }

    pub fn search(
        &self,
        start: i64,
        end: i64,
        filters: &Vec<Matchers>,
    ) -> RuntimeResult<QueryResults> {
        let inner = self.inner.read().unwrap();
        inner.search(start, end, filters)
    }
}

impl MetricDataProvider for MemoryMetricProvider {
    fn search(&self, sq: &SearchQuery, _deadline: &Deadline) -> RuntimeResult<QueryResults> {
        self.search(sq.start, sq.end, &sq.matchers)
    }
}

fn matches_filter(mn: &MetricName, filter: &LabelFilter) -> bool {
    if let Some(v) = mn.tag_value(filter.label.as_str()) {
        let fv = filter.value.as_str();
        return match filter.op {
            LabelFilterOp::Equal => v == fv,
            LabelFilterOp::NotEqual => v != fv,
            LabelFilterOp::RegexEqual => {
                let re = Regex::new(fv).unwrap();
                re.is_match(v)
            }
            LabelFilterOp::RegexNotEqual => {
                let re = Regex::new(fv).unwrap();
                !re.is_match(v)
            }
        };
    }
    false
}

fn find_first_index<'a>(range_values: &[Point], ts: i64) -> Option<usize> {
    // Find the index of the first item where `range.start <= key`.
    return match range_values.binary_search_by_key(&ts, |point| point.t) {
        Ok(index) => Some(index),

        // If the requested key is smaller than the smallest range in the slice,
        // we would be computing `0 - 1`, which would underflow an `usize`.
        // We use `checked_sub` to get `None` instead.
        Err(index) => index.checked_sub(1),
    };
}
