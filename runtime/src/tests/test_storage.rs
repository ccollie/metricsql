use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use regex::Regex;

use metricsql::prelude::{LabelFilter, LabelFilterOp};

use crate::tests::helpers::{hash_labels, Appender, Labels};
use crate::tests::value::Point;
use crate::{
    Deadline, MetricDataProvider, MetricName, QueryResults, RuntimeResult, Sample, SearchQuery,
};

pub struct TestSample {
    labels: Rc<MetricName>,
    t: i64,
    v: f64,
}

pub struct TestStorage {
    /// metric names by hash
    labels_hash: BTreeMap<u64, Rc<MetricName>>,
    sample_values: BTreeMap<u64, Vec<Point>>,
    need_sort: BTreeSet<u64>,
    pending_result: Vec<Sample>,
}

struct StorageInner {
    labels_hash: BTreeMap<u64, Rc<MetricName>>,
    sample_values: BTreeMap<u64, Vec<Point>>,
    need_sort: BTreeSet<u64>,
    pending_result: Vec<Sample>,
}

impl Appender for TestStorage {
    fn append(&mut self, labels: &Labels, t: i64, v: f64) -> RuntimeResult<()> {
        let h = hash_labels(labels);
        match self.labels_hash.entry(h) {
            Entry::Vacant(e) => {
                let mut metric = MetricName::default();
                for (k, v) in labels.iter() {
                    metric.set_tag(k, v);
                }
                let mn = Rc::new(metric);
                e.insert(Rc::clone(&mn));
                self.sample_values.insert(h, vec![Point { t, v }]);
            }
            Entry::Occupied(mut e) => {
                let values = self.sample_values.entry(h).or_default();
                values.push(Point { t, v });
            }
        }
        Ok(())
    }

    fn commit(&mut self) {
        for sample in self.pending_result.iter_mut() {
            self.add_sample(sample);
        }
    }

    fn rollback(&mut self) {
        self.pending_result.clear();
    }
}

impl TestStorage {
    pub fn new() -> Self {
        Self {
            labels_hash: Default::default(),
            sample_values: Default::default(),
            need_sort: Default::default(),
            pending_result: Default::default(),
        }
    }

    pub fn add_sample(&mut self, sample: &mut Sample) {
        let metric_id = sample.metric.get_hash();
        self.labels_hash
            .entry(metric_id)
            .or_insert_with(|| Rc::new(sample.metric.clone()));

        // todo: insert sort ?
        self.pending_result
            .entry(metric_id)
            .or_default()
            .push(Point {
                t: sample.timestamp,
                v: sample.value,
            });

        self.need_sort.insert(metric_id);
    }

    pub fn clear(&mut self) {
        self.need_sort.clear();
        self.labels_hash.clear();
        self.sample_values.clear();
    }

    pub fn search(&mut self, start: i64, end: i64, filters: &[LabelFilter]) -> Vec<TestSample> {
        let mut ids: BTreeSet<u64> = BTreeSet::new();
        let mut res: Vec<TestSample> = vec![];

        let _ = filters
            .iter()
            .for_each(|f| self.get_metric_ids_matching(f, &mut ids));

        for metric_id in ids {
            self.sort_if_needed(metric_id);
            if let Some(values) = self.sample_values.get(&metric_id) {
                if let Some(start) = find_first_index(&values, start) {
                    for point in &values[start..] {
                        if point.t > end {
                            break;
                        }
                        if let Some(labels) = self.labels_hash.get_mut(&metric_id) {
                            let sample = TestSample {
                                labels: Rc::clone(labels),
                                t: point.t,
                                v: point.v,
                            };
                            res.push(sample);
                        }
                    }
                }
            }
        }

        res
    }

    fn sort_if_needed(&mut self, id: u64) {
        if self.need_sort.contains(&id) {
            self.need_sort.remove(&id);
            if let Some(points) = self.sample_values.get_mut(&id) {
                points.sort_by(|a, b| a.t.cmp(&b.t))
            }
        }
    }

    fn get_metric_ids_matching(&self, filter: &LabelFilter, dst: &mut BTreeSet<u64>) {
        for (k, labels) in &self.labels_hash {
            if matches_filter(&labels, filter) {
                dst.insert(*k);
            }
        }
    }
}

impl MetricDataProvider for TestStorage {
    fn search(&mut self, sq: &SearchQuery, _deadline: &Deadline) -> RuntimeResult<QueryResults> {
        let temp = self.search(sq.start, sq.end, &sq.filters);
        todo!()
    }
}

fn matches_filter(mn: &MetricName, filter: &LabelFilter) -> bool {
    if let Some(v) = mn.get_tag_value(filter.label.as_str()) {
        return match filter.op {
            LabelFilterOp::Equal => v == filter.value.as_str(),
            LabelFilterOp::NotEqual => v != filter.value.as_str(),
            LabelFilterOp::RegexEqual => {
                let re = Regex::new(filter.value.as_str()).unwrap();
                re.is_match(v)
            }
            LabelFilterOp::RegexNotEqual => {
                let re = Regex::new(filter.value.as_str()).unwrap();
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
