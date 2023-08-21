use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Display;
use std::rc::Rc;
use std::time::Duration;

use crate::tests::helpers::{hash_labels, Appender, Labels};
use crate::tests::test::{test_start_time, Test};
use crate::tests::test_storage::{Point, TestStorage};
use crate::tests::types::SequenceValue;
use crate::tests::value::Point;
use crate::{MetricName, RuntimeResult};

/// load_cmd is a command that loads sequences of sample values for specific
/// metrics into the storage.
pub struct LoadCmd {
    pub(crate) gap: Duration,
    pub(crate) metrics: BTreeMap<u64, MetricName>,
    pub(crate) defs: BTreeMap<u64, Point>,
}

impl LoadCmd {
    pub fn new(gap: Duration) -> LoadCmd {
        return LoadCmd {
            gap,
            metrics: Default::default(),
            defs: Default::default(),
        };
    }

    pub(crate) fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        match self.append(t.storage) {
            Err(e) => {
                app.rollback();
                return Err(e);
            }
            Ok(_) => {
                app.commit();
                Ok(())
            }
        }
    }

    /// set a sequence of sample values for the given metric.
    pub fn set(&mut self, m: Labels, vals: &[SequenceValue]) {
        let h = hash_labels(&m);
        let mut samples: Vec<Point> = Vec::with_capacity(vals.len());
        let mut ts = test_start_time();
        for v in vals.iter() {
            if !v.omitted {
                samples.push(Point {
                    t: ts.milliseconds(),
                    v: v.value,
                })
            }
            ts += self.gap;
        }
        match self.defs.entry(h) {
            Entry::Vacant(e) => {
                for s in samples.iter_mut() {
                    s.labels = Rc::new(m.clone());
                }
                e.insert(samples);
            }
            Entry::Occupied(mut e) => {
                e.get_mut().append(&mut samples);
            }
        }
        self.defs.insert(h, samples);
        self.metrics.insert(h, m);
    }

    /// append the defined time series to the storage.
    fn append(&self, storage: &mut TestStorage) -> RuntimeResult<()> {
        for (h, samples) in self.defs.iter() {
            if let Some(m) = self.metrics.get(h) {
                for s in samples.iter_mut() {
                    storage.add_sample(s);
                }
            }
        }
        Ok(())
    }
}

impl Display for LoadCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "load {:?}", self.gap)
    }
}
