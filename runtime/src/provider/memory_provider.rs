use crate::provider::postings::PostingsEnum;
use crate::BitmapPostings;
use metricsql_parser::prelude::Matchers;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use crate::prelude::MemoryPostings;
use crate::provider::querier::postings_for_matchers;
use crate::types::MetricName;
use crate::{Deadline, MetricStorage, QueryResult, QueryResults, RuntimeError, RuntimeResult, SearchQuery};
use async_trait::async_trait;
use metricsql_common::hash::Signature;

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
#[derive(Default, Debug, Clone)]
pub struct MemoryMetricProvider {
    series: BTreeMap<Signature, (MetricName, Vec<Point>)>,
    postings: MemoryPostings,
}

impl MemoryMetricProvider {
    pub fn new() -> Self {
        MemoryMetricProvider {
            series: BTreeMap::new(),
            postings: MemoryPostings::new(),
        }
    }

    pub fn append(&mut self, labels: MetricName, t: i64, v: f64) -> RuntimeResult<()> {
        let h = labels.signature();
        let id: u64 = h.into();
        match self.series.entry(h) {
            Entry::Vacant(entry) => {
                self.postings.add_posting(id, &labels);
                entry.insert((labels, vec![Point { t, v }]));
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().1.push(Point { t, v });
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.series.clear();
        self.postings.clear();
    }

    // Get the series data without any async operations
    pub fn get_series_data(&self, signature: &Signature) -> Option<(MetricName, Vec<Point>)> {
        self.series.get(signature).cloned()
    }

    pub async fn get_postings_for_matchers(&self, matchers: &Matchers) -> RuntimeResult<PostingsEnum<BitmapPostings>> {
        postings_for_matchers(&self.postings, matchers).await
            .map_err(|e| RuntimeError::ProviderError(e.to_string()))
    }

    async fn search_internal(&self, sq: SearchQuery) -> RuntimeResult<QueryResults> {
        // Now we can safely call the async function without holding the lock
        let postings = self.get_postings_for_matchers(&sq.matchers).await
            .map_err(|e| RuntimeError::ProviderError(e.to_string()))?;

        let start = sq.start;
        let end = sq.end;
        let mut results: Vec<QueryResult> = vec![];

        // Collect the IDs to process
        let ids: Vec<u64> = postings.collect();

        // Process each ID, taking and releasing the lock for each one to avoid holding it across awaits
        for id in ids {
            let signature = Signature::from(id);
            let data_option = {
                self.get_series_data(&signature)
            };

            if let Some((metric_name, data)) = data_option {
                if let Some(first_idx) = find_first_index(&data, start) {
                    let mut last_idx = first_idx;
                    while last_idx < data.len() {
                        if data[last_idx].t > end {
                            break;
                        }
                        last_idx += 1;
                    }
                    let samples = &data[first_idx..=last_idx.min(data.len() - 1)]; // Added bounds check
                    let values = samples.iter().map(|x| x.v).collect();
                    let timestamps = samples.iter().map(|x| x.t).collect();
                    results.push(QueryResult {
                        metric: metric_name.clone(),
                        values,
                        timestamps,
                    });
                }
            }
        }

        Ok(QueryResults::new(results))
    }
}

#[async_trait]
impl MetricStorage for MemoryMetricProvider {
    async fn search(&self, sq: SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        self.search_internal(sq).await
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
        let mut provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();

        let signature = labels.signature();
        let id: u64 = signature.into();
        assert!(provider.postings.has_posting(id));
    }

    #[test]
    fn append_existing_metric_adds_point() {
        let mut provider = MemoryMetricProvider::new();
        let mut labels = MetricName::default();
        labels.add_label("foo", "bar");
        provider.append(labels.clone(), 1, 1.0).unwrap();
        provider.append(labels.clone(), 2, 2.0).unwrap();

        let signature = labels.signature();
        if let Some((_, data)) = provider.series.get(&signature) {
            assert_eq!(data.len(), 2);
        } else {
            panic!("No data found for metric: {:?}", labels);
        }
    }
}
