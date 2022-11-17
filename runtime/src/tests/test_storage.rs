use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

struct TestSample {
    labels: Rc<MetricName>,
    t: u64,
    v: f64
}

pub struct TestStorage {
    // metric names by hash
    labels_hash: BTreeMap<u64, Rc<MetricName>>,
    sample_values: BTreeMap<u64, Vec<Point>>,
    need_sort: BTreeSet<u64>
}

impl TestStorage {
    pub fn add_sample(&mut self, sample: &Sample) {
        let metric_id = sample.metric_name.hash();
        self.labels_hash.entry(metric_id).or_insert_with(|x| {
            Rc::new(sample.metric_name.clone())
        });

        // todo: insert sort ?
        self.sample_values.entry(metric_id)
            .or_default()
            .push(Point {
                t: sample.t,
                v: sample.v
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

        filters.for_each(|f| self.get_metric_ids_matching(f, ids));
        for metric_id in ids {
            match (self.sample_values.get(metric_id), self.labels_hash.get_mut(metric_id)) {
                (Some(values), Some(labels)) => {
                    self.sort_if_needed(metric_id);
                    if let Some(start) = find_first_index(&bucket, start) {
                        let mut i = start;
                        while i < values.len() {
                            let point = values[i];
                            if point.t > end {
                                break;
                            }
                            let sample = TestSample {
                                labels: Rc::clone(labels),
                                t: point.t,
                                v: point.v
                            };
                            res.push(sample);
                            i += 1;
                        }
                    }
                },
                _ => continue
            }
        }

        res
    }

    fn sort_if_needed(&mut self, id: u64) {
        if self.need_sort(id) {
            self.need_sort.remove(id);
            if let Some(points) = self.sample_values.get_mut(id) {
                points.sort_by(|a, b| a.t.cmp(b.t) )
            }
        }
    }

    fn get_metric_ids_matching(&self, filter: &LabelFilter, dst: BTreeSet<u64>) {
        for (k, labels) in self.labels_hash {
          if matches_filter(labels, filter) {
              dst.insert(k);
          }
        }
    }
}

fn matches_filter(mn: &MetricName, filter: LabelFilter) -> bool {
    todo!()
}


fn find_first_index<'a>(range_values: &[Point], ts: i64) -> Option<usize> {
    // Find the index of the first item where `range.start <= key`.
    return
        match range_values.binary_search_by_key(ts, |point| point.t) {
            Ok(index) => Some(index),

            // If the requested key is smaller than the smallest range in the slice,
            // we would be computing `0 - 1`, which would underflow an `usize`.
            // We use `checked_sub` to get `None` instead.
            Err(index) => index.checked_sub(1),
        };
}