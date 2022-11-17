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


use std::fmt::{Display, Formatter};
use std::task::Context;

pub(crate) struct NoopAppendable{}
pub struct NoopAppender{}


impl Appendable for NoopAppendable {
    fn get_appender(_: Context) -> Appender {
        return NoopAppender{}
    }
}

impl Appender for NoopAppender {
    fn append(&self, _sref: SeriesRef, _labels: Labels, _ts: i64, _v: f64) -> RuntimeResult<SeriesRef> {
        return Ok(0)
    }
    
    fn commit(&self) -> RuntimeResult<()> { 
        Ok(()) 
    }

    fn rollback(&self) -> RuntimeResult<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    metric: MetricName,
    t: i64,
    v: f64
}

impl Sample {
    pub fn new(labels: MetricName, t: i64, v: f64) -> Self {
        Self {
            metric: labels.clone(),
            t,
            v
        }
    }    
}

/// CollectResultAppender records all samples that were added through the appender.
/// It can be used as its zero value or be backed by another appender it writes samples through.
#[derive(Default, Clone)]
pub struct CollectResultAppender {
    next: Option<Box<dyn Appender>>,
    result : Vec<Sample>,
    pending_result: Vec<Sample>,
    rolledback_result: Vec<Sample>,
}

impl Appender for CollectResultAppender {
    fn append(&mut self, sref: SeriesRef, l: Labels, t: i64, v: f64) -> RuntimeResult<SeriesRef> {
        self.pending_result.push(Sample::new(l, t, v));
        let res = 0;
        if _ref == 0 {
            res = storage.SeriesRef(rand.Uint64())
        } else {
            res = sref
        }
        if self.next.is_none() {
            return Ok(res)
        }
        
        self.next.unwrap().append(res, lset, t, v)
    }
    
    fn commit(&mut self) -> RuntimeResult<()> {
        self.result.extend_from_slice(self.pending_result.into_iter());
        self.pending_result.clear();
        if let Some(next) = self.next {
            return self.next.commit();
        } else {
            return Ok(())
        }
    }

    fn rollback(&mut self) -> RuntimeResult<()> {
        self.rolledback_result = std::mem::take(&mut self.pending_result);
        if let Some(next) = self.next {
            return next.rollback();
        }
        Ok(())        
    }
}

impl Display for CollectResultAppender {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for s in self.result.iter()  {
            write!(f, "committed: {} {} {}\n", s.metric, s.v, s.t);
        }
        for s in self.pending_result.iter()  {
            write!(f, "pending: {} {} {}\n", s.metric, s.v, s.t);
        }
        for s in self.rolledback_result.iter()  {
            write!(f, "rolled back: {} {} {}\n", s.metric, s.v, s.t);
        }
    }
}
