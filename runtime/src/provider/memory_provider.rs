use crate::prelude::{MemoryPostings, Sample};
use crate::provider::postings::PostingsEnum;
use crate::provider::querier::postings_for_matchers;
use crate::types::{MetricName, Timestamp};
use crate::BitmapPostings;
use crate::{
    Deadline, MetricStorage, QueryResult, QueryResults, RuntimeError, RuntimeResult, SearchQuery,
};
use async_trait::async_trait;
use metricsql_common::hash::Signature;
use metricsql_parser::prelude::Matchers;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::sync::RwLock;

struct StorageInner {
    series: BTreeMap<Signature, (MetricName, Vec<Sample>)>,
}

struct MemoryStorage {
    series: RwLock<StorageInner>,
}

impl MemoryStorage {
    fn new() -> Self {
        MemoryStorage {
            series: RwLock::new(StorageInner {
                series: BTreeMap::new(),
            }),
        }
    }

    fn clear(&self) {
        let mut inner = self.series.write().unwrap();
        inner.series.clear();
    }

    fn append(&self, labels: &MetricName, t: i64, v: f64) -> RuntimeResult<bool> {
        let h = labels.signature();
        let mut inner = self.series.write().unwrap();
        let sample = Sample::new(t, v);
        match inner.series.entry(h) {
            Entry::Vacant(entry) => {
                entry.insert((labels.clone(), vec![sample]));
                Ok(true)
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().1.push(sample);
                Ok(false)
            }
        }
    }

    pub fn get_series_data(
        &self,
        signature: &Signature,
        start: Timestamp,
        end: Timestamp,
    ) -> Option<QueryResult> {
        let inner = self.series.read().unwrap();

        if let Some((metric_name, data)) = inner.series.get(signature) {
            if let Some(first_idx) = find_first_index(data, start) {
                let mut last_idx = first_idx;
                while last_idx < data.len() {
                    if data[last_idx].timestamp > end {
                        break;
                    }
                    last_idx += 1;
                }
                let samples = &data[first_idx..=last_idx.min(data.len() - 1)]; // Added bounds check
                let values = samples.iter().map(|x| x.value).collect();
                let timestamps = samples.iter().map(|x| x.timestamp).collect();
                return Some(QueryResult {
                    metric: metric_name.clone(),
                    values,
                    timestamps,
                });
            }
        }
        None
    }
}

/// In-memory implementation of MetricDataProvider primarily for testing
pub struct MemoryMetricProvider {
    series: MemoryStorage,
    postings: MemoryPostings,
}

impl MemoryMetricProvider {
    pub fn new() -> Self {
        MemoryMetricProvider {
            series: MemoryStorage::new(),
            postings: MemoryPostings::new(),
        }
    }

    pub fn append(&self, labels: &MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let h = labels.signature();
        let id: u64 = h.into();
        if self.series.append(labels, t, v)? {
            // If this is a new metric, we need to add it to the postings
            self.postings.add_posting(id, labels);
        }
        Ok(())
    }

    pub fn clear(&self) {
        self.series.clear();
        self.postings.clear();
    }

    // Get the series data without any async operations
    pub fn get_series_data(
        &self,
        signature: &Signature,
        start: Timestamp,
        end: Timestamp,
    ) -> Option<QueryResult> {
        self.series.get_series_data(signature, start, end)
    }

    pub async fn get_postings_for_matchers(
        &self,
        matchers: &Matchers,
    ) -> RuntimeResult<PostingsEnum<BitmapPostings>> {
        postings_for_matchers(&self.postings, matchers)
            .await
            .map_err(|e| RuntimeError::ProviderError(e.to_string()))
    }

    fn get_data_for_postings(
        &self,
        start: Timestamp,
        end: Timestamp,
        postings: PostingsEnum<BitmapPostings>,
    ) -> Vec<QueryResult> {
        postings
            .into_iter()
            .flat_map(|posting| {
                let id: u64 = posting;
                let signature = Signature::from(id);
                self.series.get_series_data(&signature, start, end)
            })
            .collect()
    }

    async fn search_internal(&self, sq: SearchQuery) -> RuntimeResult<QueryResults> {
        let postings = self
            .get_postings_for_matchers(&sq.matchers)
            .await
            .map_err(|e| RuntimeError::ProviderError(e.to_string()))?;

        let results = self.get_data_for_postings(sq.start, sq.end, postings);

        Ok(QueryResults::new(results))
    }
}

impl Default for MemoryMetricProvider {
    fn default() -> Self {
        MemoryMetricProvider::new()
    }
}

#[async_trait]
impl MetricStorage for MemoryMetricProvider {
    async fn search(&self, sq: SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        self.search_internal(sq).await
    }
}

fn find_first_index(range_values: &[Sample], ts: i64) -> Option<usize> {
    // Find the index of the first item where `range.start <= key`.
    match range_values.binary_search_by_key(&ts, |point| point.timestamp) {
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
        provider.append(&labels, 1, 1.0).unwrap();

        let signature = labels.signature();
        let id: u64 = signature.into();
        assert!(provider.postings.has_posting(id));
    }

    #[test]
    fn append_existing_metric_adds_point() {
        let mut provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(&labels, 1, 1.0).unwrap();
        provider.append(&labels, 2, 2.0).unwrap();

        let signature = labels.signature();
        if let Some(data) = provider.series.get_series_data(&signature, 0, 10) {
            assert_eq!(data.len(), 2);
        } else {
            panic!("No data found for metric: {:?}", labels);
        }
    }
}
