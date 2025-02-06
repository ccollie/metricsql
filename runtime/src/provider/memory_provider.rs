use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::RwLock;

use async_trait::async_trait;
use metricsql_common::hash::Signature;
use metricsql_parser::prelude::{Matcher, Matchers};

use crate::prelude::MemoryPostings;
use crate::types::MetricName;
use crate::{
    Deadline, 
    MetricStorage,
    QueryResult,
    QueryResults,
    RuntimeError,
    RuntimeResult,
    SearchQuery
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
    series: BTreeMap<Signature, (MetricName, Vec<Point>)>,
    postings: MemoryPostings
}

impl Storage {
    pub fn append(&mut self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let h = labels.signature();
        let id: u64 = h.into();
        match self.series.entry(h) {
            Entry::Vacant(entry) => {
                self.postings.add_posting(id, &labels);
                entry.insert((labels, vec![Point { t, v }]));
            },
            Entry::Occupied(mut entry) => {
                entry.get_mut().1.push(Point { t, v });
            },
        }
        Ok(())
    }

    pub fn search(&self, start: i64, end: i64, filters: &Matchers) -> RuntimeResult<QueryResults> {
        let mut results: Vec<QueryResult> = vec![];
        let found = self.postings.postings_for_matchers(filters)
            .map_err(|_| RuntimeError::ProviderError(filters.to_string()))?;

        for id in found.iter() {
            let signature = Signature::from(id);
            if let Some((metric_name, data)) = self.series.get(&signature) {
                if let Some(first_idx) = find_first_index(&data, start) {
                    let mut last_idx = first_idx;
                    while last_idx < data.len() {
                        if data[last_idx].t > end {
                            break;
                        }
                        last_idx += 1;
                    }
                    let samples = &data[first_idx..=last_idx];
                    let values = samples.iter().map(|x| x.v).collect();
                    let timestamps = samples.iter().map(|x| x.t).collect();
                    results.push(QueryResult {
                        metric: metric_name.clone(),
                        values,
                        timestamps
                    });
                }
            }
        }

        Ok(QueryResults::new(results))
    }

    fn clear(&mut self) {
        self.series.clear();
        self.postings.clear();
    }
}

impl MemoryMetricProvider {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Storage {
                series: Default::default(),
                postings: MemoryPostings::new(),
            }),
        }
    }

    pub fn append(&self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(labels, t, v)
    }

    pub fn add_sample(&self, sample: Sample) -> RuntimeResult<()> {
        let mut inner = self.inner.write().unwrap();
        inner.append(sample.metric, sample.timestamp, sample.value)
    }

    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.clear();
    }

    fn search_internal(
        &self,
        start: i64,
        end: i64,
        filters: &Matchers,
    ) -> RuntimeResult<QueryResults> {
        let inner = self.inner.read().unwrap();
        inner.search(start, end, filters)
    }

    pub fn search(&self, start: i64, end: i64, filters: &Matchers) -> RuntimeResult<QueryResults> {
        self.search_internal(start, end, filters)
    }
}

#[async_trait]
impl MetricStorage for MemoryMetricProvider {
    async fn search(&self, sq: SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        self.search_internal(sq.start, sq.end, &sq.matchers)
    }
}


fn find_first_index(range_values: &[Point], ts: i64) -> Option<usize> {
    // Find the index of the first item where `range.start <= key`.
    match range_values.binary_search_by_key(&ts, |point| point.t) {
        Ok(index) => Some(index),

        // If the requested key is smaller than the smallest range in the slice,
        // we would be computing `0 - 1`, which would underflow an `usize`.
        // We use `checked_sub` to get `None` instead.
        Err(index) => index.checked_sub(1),
    }
}

#[cfg(test)]
mod tests {
    use crate::types::MetricName;

    use super::*;

    #[test]
    fn append_new_metric_creates_new_entry() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let signature = labels.signature();
        let id: u64 = signature.into();
        let inner = provider.inner.read().unwrap();
        assert!(inner.postings.has_posting(id));
    }

    #[test]
    fn append_existing_metric_adds_point() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();
        provider.append(labels.clone(), 2, 2.0).unwrap();

        let inner = provider.inner.read().unwrap();
        
        let signature = labels.signature();
        if let Some((_, data)) = inner.series.get(&signature) {
            assert_eq!(data.len(), 2);
        } else {
            panic!("No data found for metric: {:?}", labels);
        }
    }

    #[test]
    fn search_returns_matching_metrics() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let matchers = Matchers::new(vec![Matcher::equal("foo", "bar")]);
        let results = provider.search(0, 2, &matchers).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_returns_empty_for_no_match() {
        let provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let matchers = Matchers::new(vec![Matcher::equal("foo", "baz")]);
        let results = provider.search(0, 2, &matchers).unwrap();

        assert_eq!(results.len(), 0);
    }
}
