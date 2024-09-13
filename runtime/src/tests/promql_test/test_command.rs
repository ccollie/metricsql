// Copyright 2015 The Prometheus Authors
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
use super::types::{Sample, SequenceValue};
use super::utils::{almost_equal, assert_matrix_sorted, format_series_result, DEFAULT_EPSILON};
use crate::signature::Signature;
use crate::{MemoryMetricProvider, MetricName, QueryValue};
use ahash::{HashSet, HashSetExt};
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};


// Clear command
#[derive(Debug)]
pub struct ClearCmd;
impl Display for ClearCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "clear")
    }
}

// Load command
#[derive(Debug)]
pub(crate) struct LoadCmd {
    pub(super) gap: Duration,
    pub(super) metrics: HashMap<Signature, MetricName>,
    pub(super) defs: HashMap<Signature, Vec<Sample>>,
    with_nhcb: bool,
}

impl Display for LoadCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "load")
    }
}

impl LoadCmd {
    pub(crate) fn new(gap: Duration, with_nhcb: bool) -> Self {
        Self {
            gap,
            metrics: HashMap::new(),
            defs: HashMap::new(),
            with_nhcb,
        }
    }

    pub(crate) fn set(&mut self, m: MetricName, vals: Vec<SequenceValue>) {
        let hash = m.signature();
        let mut samples = Vec::with_capacity(vals.len());
        let mut ts = SystemTime::UNIX_EPOCH;
        for v in vals.iter() {
            if !v.omitted {
                samples.push(Sample {
                    metric: m.clone(),
                    timestamp: ts.duration_since(UNIX_EPOCH).unwrap().as_millis() as i64,
                    value: v.value,
                });
            }
            ts += self.gap;
        }
        self.defs.insert(hash, samples);
        self.metrics.insert(hash, m);
    }

    // append the defined time series to the storage.
    pub(super) fn append(&mut self, storage: &Arc<MemoryMetricProvider>) {
        for (h, smpls) in self.defs.iter() {
            if let Some(m) = self.metrics.get(h) {
                for s in smpls.iter() {
                    storage.append(m.clone(), s.timestamp, s.value).unwrap();
                }
            }
        }
    }
}

// Eval command
#[derive(Debug)]
pub(crate) struct EvalCmd {
    pub expr: String,
    pub start: SystemTime,
    pub end: SystemTime,
    pub step: Duration,
    pub line: usize,
    pub is_range: bool,
    pub fail: bool,
    pub warn: bool,
    pub ordered: bool,
    pub expected_fail_message: Option<String>,
    pub expected_fail_regexp: Option<Regex>,
    pub metrics: HashMap<Signature, MetricName>,
    pub expect_scalar: bool,
    pub expected: HashMap<Signature, Entry>,
}

impl Display for EvalCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval({})", self.expr)
    }
}

#[derive(Debug)]
struct Entry {
    pos: usize,
    vals: Vec<SequenceValue>,
}

static SCALAR_SIGNATURE: Signature = Signature(0);

impl EvalCmd {
    pub(crate) fn new_instant_eval_cmd(expr: String, start: SystemTime, line: usize) -> Self {
        Self {
            expr,
            start,
            end: start,
            step: Duration::from_secs(0),
            line,
            is_range: false,
            fail: false,
            warn: false,
            ordered: false,
            expected_fail_message: None,
            expected_fail_regexp: None,
            metrics: HashMap::new(),
            expect_scalar: false,
            expected: HashMap::new(),
        }
    }

    pub(crate) fn new_range_eval_cmd(expr: String, start: SystemTime, end: SystemTime, step: Duration, line: usize) -> Self {
        Self {
            expr,
            start,
            end,
            step,
            line,
            is_range: true,
            fail: false,
            warn: false,
            ordered: false,
            expected_fail_message: None,
            expected_fail_regexp: None,
            metrics: HashMap::new(),
            expect_scalar: false,
            expected: HashMap::new(),
        }
    }

    pub(crate) fn expect(&mut self, pos: usize, vals: Vec<SequenceValue>) {
        self.expect_scalar = true;
        self.expected.insert(SCALAR_SIGNATURE, Entry { pos, vals });
    }

    pub(crate) fn expect_metric(&mut self, pos: usize, m: MetricName, vals: Vec<SequenceValue>) {
        self.expect_scalar = false;
        let hash = m.signature();
        self.metrics.insert(hash, m);
        self.expected.insert(hash, Entry { pos, vals });
    }

    pub(super) fn compare_result(&self, result: &QueryValue) -> Result<(), Box<dyn Error>> {
        match result {
            QueryValue::Scalar(val) => {
                if !self.expect_scalar {
                    return Err(format!("expected vector or matrix result, but got {}", val).into());
                }
                if let Some(exp0) = self.expected.get(&SCALAR_SIGNATURE) {
                    if !almost_equal(exp0.vals[0].value, *val, DEFAULT_EPSILON) {
                        return Err(format!("expected scalar {:?} but got {}", exp0.vals[0].value, val).into());
                    }
                }
            }
            QueryValue::InstantVector(val) => {
                if self.expect_scalar {
                    return Err(format!("expected scalar result, but got vector {}", val).into());
                }

                let mut seen = HashSet::with_capacity(self.expected.len());
                for (pos, v) in val.iter().enumerate() {
                    let f = v.values[0];
                    let fp = v.metric_name.signature();
                    if !self.metrics.contains_key(&fp) {
                        return Err(format!("unexpected metric {} in result, has value {}",
                                           v.metric_name, f).into());
                    }
                    let exp = &self.expected[&fp];
                    if self.ordered && exp.pos != pos + 1 {
                        return Err(format!("expected metric {} with {:?} at position {} but was at {}",
                                           v.metric_name, exp.vals, exp.pos, pos + 1).into());
                    }
                    let exp0 = &exp.vals[0];
                    if !almost_equal(exp0.value, f, DEFAULT_EPSILON) {
                        return Err(format!("expected {:?} for {} but got {f}",
                                           exp0.value, v.metric_name).into());
                    }

                    seen.insert(fp);
                }
                for fp in self.expected.keys() {
                    if !seen.contains(fp) {
                        return Err(format!("expected metric {} with {:?} not found",
                                           self.metrics[fp],
                                           self.expected[fp]).into());
                    }
                }
            }
            QueryValue::RangeVector(val) => {
                if self.ordered {
                    return Err("expected ordered result, but query returned a matrix".to_string().into());
                }

                if self.expect_scalar {
                    return Err(format!("expected scalar result, but got matrix {}", val).into());
                }

                if let Err(err) = assert_matrix_sorted(result) {
                    return Err(format!("expected sorted matrix result, but got unsorted matrix: {}", err).into());
                }

                let mut seen = HashMap::new();
                for s in val.iter() {
                    let hash = s.metric_name.signature();
                    if !self.metrics.contains_key(&hash) {
                        return Err(format!("unexpected metric {} in result, has {}",
                                           s.metric_name, format_series_result(s)).into());
                    }
                    seen.insert(hash, true);
                    let exp = &self.expected[&hash];

                    let mut expected_floats = Vec::new();

                    let step_millis = self.step.as_millis() as u64; // do we have negative steps ?????
                    for (i, e) in exp.vals.iter().enumerate() {
                        let ts = self.start + Duration::from_millis((i as u64 * step_millis));

                        if ts > self.end {
                            return Err(format!("expected {} points for {}, but query time range cannot return this many points", exp.vals.len(), self.metrics[&hash]).into());
                        }

                        let timestamp = ts.duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;

                        if !e.omitted {
                            expected_floats.push(Sample { timestamp, value: e.value, metric: s.metric_name.clone() });
                        }
                    }

                    if expected_floats.len() != s.values.len() {
                        return Err(format!("expected {} float points for {}, but got {}", expected_floats.len(),
                                           self.metrics[&hash], format_series_result(s)).into());
                    }

                    for (i, expected) in expected_floats.iter().enumerate() {
                        let timestamp = &s.timestamps[i];
                        let value = &s.values[i];

                        if expected.timestamp != *timestamp {
                            return Err(format!("expected float value at index {} for {} to have timestamp {}, but it had timestamp {} (result has {})", i, self.metrics[&hash],
                                               expected.timestamp, timestamp, format_series_result(s)).into());
                        }

                        if !almost_equal(*value, expected.value, DEFAULT_EPSILON) {
                            return Err(format!("expected float value at index {} (t={}) for {} to be {}, but got {} (result has {})", i,
                                               timestamp,
                                               self.metrics[&hash],
                                               expected.value,
                                               value,
                                               format_series_result(s)).into());
                        }
                    }
                }

                for hash in self.expected.keys() {
                    if !seen.contains_key(hash) {
                        return Err(format!("expected metric {} not found", self.metrics[hash]).into());
                    }
                }
            }
            None => {}
        }

        Ok(())
    }
}


#[derive(Debug)]
pub enum TestCommand {
    Clear(ClearCmd),
    Eval(EvalCmd),
    Load(LoadCmd),
}