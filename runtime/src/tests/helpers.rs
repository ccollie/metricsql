// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{MetricName, RuntimeResult};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::task::Context;

pub type Labels = HashMap<String, String>;

/// Appender provides batched appends against a storage.
/// It must be completed with a call to Commit or Rollback and must not be reused afterwards.
///
/// Operations on the Appender interface are not goroutine-safe.
///
/// The type of samples (float64, histogram, etc) appended for a given series must remain same within an Appender.
/// The behaviour is undefined if samples of different types are appended to the same series in a single Commit().
pub trait Appender {
    /// Append adds a sample pair for the given series.
    /// An optional series reference can be provided to accelerate calls.
    /// A series reference number is returned which can be used to add further
    /// samples to the given series in the same or later transactions.
    /// Returned reference numbers are ephemeral and may be rejected in calls
    /// to append() at any point. Adding the sample via Append() returns a new
    /// reference number.
    /// If the reference is 0 it must not be used for caching.
    fn append(&mut self, l: Labels, t: i64, v: f64) -> RuntimeResult<()>;

    /// Commit submits the collected samples and purges the batch. If Commit
    /// returns a non-nil error, it also rolls back all modifications made in
    /// the appender so far, as Rollback would do. In any case, an Appender
    /// must not be used anymore after Commit has been called.
    fn commit(&mut self) -> RuntimeResult<()>;

    /// rollback rolls back all modifications made in the appender so far.
    /// Appender has to be discarded after rollback.
    fn rollback(&mut self) -> RuntimeResult<()>;
}

pub trait Appendable {
    fn get_appender(_: Context) -> Box<dyn Appender> {
        return Box::new(NoopAppender {});
    }
}

pub(crate) struct NoopAppendable {}
pub struct NoopAppender {}

impl Appendable for NoopAppendable {
    fn get_appender(_: Context) -> Box<dyn Appender> {
        return Box::new(NoopAppender {});
    }
}

impl Appender for NoopAppender {
    fn append(&mut self, _labels: Labels, _ts: i64, _v: f64) -> RuntimeResult<()> {
        return Ok(());
    }

    fn commit(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    fn rollback(&mut self) -> RuntimeResult<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    metric: MetricName,
    t: i64,
    v: f64,
}

impl Sample {
    pub fn new(labels: MetricName, t: i64, v: f64) -> Self {
        Self {
            metric: labels,
            t,
            v,
        }
    }

    pub fn from_hashmap(map: &HashMap<String, String>, t: i64, v: f64) -> Self {
        let mut metric_name = MetricName::new("");
        for (k, v) in map.iter() {
            metric_name.set_tag(k.as_str(), v)
        }
        Self {
            metric: metric_name,
            t,
            v,
        }
    }
}

/// CollectResultAppender records all samples that were added through the appender.
/// It can be used as its zero value or be backed by another appender it writes samples through.
#[derive(Default)]
pub struct CollectResultAppender {
    result: Vec<Sample>,
    pending_result: Vec<Sample>,
    rolledback_result: Vec<Sample>,
}

impl CollectResultAppender {
    pub fn new() -> Self {
        Self {
            result: vec![],
            pending_result: vec![],
            rolledback_result: vec![],
        }
    }
}

impl Appender for CollectResultAppender {
    fn append(&mut self, l: Labels, t: i64, v: f64) -> RuntimeResult<()> {
        self.pending_result.push(Sample::from_hashmap(&l, t, v));
        Ok(())
    }

    fn commit(&mut self) -> RuntimeResult<()> {
        self.result.extend_from_slice(&self.pending_result.clone());
        self.pending_result.clear();
        Ok(())
    }

    fn rollback(&mut self) -> RuntimeResult<()> {
        self.rolledback_result = std::mem::take(&mut self.pending_result);
        Ok(())
    }
}

impl Display for CollectResultAppender {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for s in self.result.iter() {
            write!(f, "committed: {} {} {}\n", s.metric, s.v, s.t)?;
        }
        for s in self.pending_result.iter() {
            write!(f, "pending: {} {} {}\n", s.metric, s.v, s.t)?;
        }
        for s in self.rolledback_result.iter() {
            write!(f, "rolled back: {} {} {}\n", s.metric, s.v, s.t)?;
        }
        Ok(())
    }
}
