// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License apub t
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use super::types::{is_info, is_warning, Annotations, ExpectCmd, ExpectCmdType, Sample, SequenceValue, TestAssertionError};
use super::utils::{almost_equal, assert_matrix_sorted, format_series_result, DEFAULT_EPSILON};
use crate::postings::PostingsEnum;
use crate::querier::postings_for_matchers;
use crate::types::{MetricName, QueryValue};
use crate::{BitmapPostings, MemoryMetricProvider, MemoryPostings, RuntimeError, RuntimeResult};
use ahash::{HashSet, HashSetExt};
use metricsql_common::hash::Signature;
use metricsql_parser::label::Matchers;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::convert::Into;
use std::fmt;
use std::fmt::Display;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Clear command
#[derive(Debug, Clone)]
pub struct ClearCmd;
impl Display for ClearCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "clear")
    }
}

// Load command
#[derive(Debug, Clone)]
pub(crate) struct LoadCmd {
    pub(super) gap: Duration,
    pub(super) metrics: HashMap<Signature, MetricName>,
    pub(super) defs: HashMap<Signature, Vec<Sample>>,
    series: BTreeMap<Signature, (MetricName, Vec<Sample>)>,
    postings: MemoryPostings,
}

impl Display for LoadCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "load")
    }
}

impl LoadCmd {
    pub(crate) fn new(gap: Duration) -> Self {
        Self {
            gap,
            metrics: HashMap::new(),
            defs: HashMap::new(),
            series: Default::default(),
            postings: MemoryPostings::new(),
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
    pub(super) fn append(&self, storage: &Arc<MemoryMetricProvider>) {
        for (h, smpls) in self.defs.iter() {
            if let Some(m) = self.metrics.get(h) {
                for s in smpls.iter() {
                    storage.append(m, s.timestamp, s.value).unwrap();
                }
            }
        }
    }

    async fn get_postings_for_matchers(
        &self,
        matchers: &Matchers,
    ) -> RuntimeResult<PostingsEnum<BitmapPostings>> {
        postings_for_matchers(&self.postings, matchers)
            .await
            .map_err(|e| RuntimeError::ProviderError(e.to_string()))
    }
}

// Eval command
#[derive(Debug, Clone)]
pub(crate) struct EvalCmd {
    pub expr: String,
    pub start: SystemTime,
    pub end: SystemTime,
    pub step: Duration,
    pub line: usize,
    pub is_range: bool,
    pub fail: bool,
    pub warn: bool,
    pub info: bool,
    pub ordered: bool,
    pub expected_fail_message: Option<String>,
    pub expected_fail_regexp: Option<Regex>,
    pub metrics: HashMap<Signature, MetricName>,
    pub expect_scalar: bool,
    pub expected: HashMap<Signature, Entry>,
    pub expected_cmds: HashMap<ExpectCmdType, Vec<ExpectCmd>>,
}

impl Display for EvalCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval({})", self.expr)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pos: usize,
    vals: Vec<SequenceValue>,
}

static SCALAR_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| 0.into());

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
            info: false,
            ordered: false,
            expected_fail_message: None,
            expected_fail_regexp: None,
            metrics: HashMap::new(),
            expect_scalar: false,
            expected: HashMap::new(),
            expected_cmds: Default::default(),
        }
    }

    pub(crate) fn new_range_eval_cmd(
        expr: String,
        start: SystemTime,
        end: SystemTime,
        step: Duration,
        line: usize,
    ) -> Self {
        Self {
            expr,
            start,
            end,
            step,
            line,
            is_range: true,
            fail: false,
            warn: false,
            info: false,
            ordered: false,
            expected_fail_message: None,
            expected_fail_regexp: None,
            metrics: HashMap::new(),
            expect_scalar: false,
            expected: HashMap::new(),
            expected_cmds: Default::default(),
        }
    }

    pub fn is_ordered(&self) -> bool {
        if self.ordered {
            return true;
        }
        match self.expected_cmds.get(&ExpectCmdType::Ordered) {
            Some(cmds) => !cmds.is_empty(),
            None => false,
        }
    }

    pub fn is_fail(&self) -> bool {
        if self.fail {
            return true;
        }
        match self.expected_cmds.get(&ExpectCmdType::Fail) {
            Some(cmds) => !cmds.is_empty(),
            None => false,
        }
    }

    pub(crate) fn expect(&mut self, pos: usize, vals: Vec<SequenceValue>) {
        self.expect_scalar = true;
        self.expected.insert(*SCALAR_SIGNATURE, Entry { pos, vals });
    }

    pub(crate) fn expect_metric(&mut self, pos: usize, m: MetricName, vals: Vec<SequenceValue>) {
        self.expect_scalar = false;
        let hash = m.signature();
        self.metrics.insert(hash, m);
        self.expected.insert(hash, Entry { pos, vals });
    }

    pub(super) fn compare_result(&self, result: &QueryValue) -> Result<(), TestAssertionError> {
        match result {
            QueryValue::Scalar(val) => {
                if !self.expect_scalar {
                    let msg = format!("expected vector or matrix result, but got {}", val);
                    return Err(TestAssertionError::new(self.line, msg));
                }
                if let Some(exp0) = self.expected.get(&SCALAR_SIGNATURE) {
                    if !almost_equal(exp0.vals[0].value, *val, DEFAULT_EPSILON) {
                        let msg = format!("expected {:?} but got {}", exp0.vals[0].value, val);
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                }
            }
            QueryValue::InstantVector(val) => {
                if self.expect_scalar {
                    let msg = format!("expected scalar result, but got vector {:?}", val);
                    return Err(TestAssertionError::new(self.line, msg));
                }

                let mut seen = HashSet::with_capacity(self.expected.len());
                for (pos, v) in val.iter().enumerate() {
                    let f = v.values[0];
                    let fp = v.metric_name.signature();
                    if !self.metrics.contains_key(&fp) {
                        let msg = format!(
                            "unexpected metric {} in result, has value {}",
                            v.metric_name, f
                        );
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                    let exp = &self.expected[&fp];
                    if self.ordered && exp.pos != pos + 1 {
                        let msg = format!(
                            "expected metric {} with {:?} at position {} but was at {}",
                            v.metric_name,
                            exp.vals,
                            exp.pos,
                            pos + 1
                        );
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                    let exp0 = &exp.vals[0];
                    if !almost_equal(exp0.value, f, DEFAULT_EPSILON) {
                        let msg = format!(
                            "expected {:?} for {} but got {f}",
                            exp0.value, v.metric_name
                        );

                        return Err(TestAssertionError::new(self.line, msg));
                    }

                    seen.insert(fp);
                }
                for fp in self.expected.keys() {
                    if !seen.contains(fp) {
                        let msg = format!(
                            "expected metric {} with {:?} not found",
                            self.metrics[fp], self.expected[fp]
                        );
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                }
            }
            QueryValue::RangeVector(val) => {
                if self.is_ordered() {
                    let msg = "expected ordered result, but query returned a matrix".to_string();
                    return Err(TestAssertionError::new(self.line, msg));
                }

                if self.expect_scalar {
                    let msg = format!("expected scalar result, but got matrix {:?}", val);
                    return Err(TestAssertionError::new(self.line, msg));
                }

                if let Err(err) = assert_matrix_sorted(result) {
                    let msg = format!(
                        "expected sorted matrix result, but got unsorted matrix: {:?}",
                        err
                    );
                    return Err(TestAssertionError::new(self.line, msg));
                }

                let mut seen = HashMap::new();
                for s in val.iter() {
                    let hash = s.metric_name.signature();
                    if !self.metrics.contains_key(&hash) {
                        let msg = format!(
                            "unexpected metric {} in result, has {}",
                            s.metric_name,
                            format_series_result(s)
                        );
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                    seen.insert(hash, true);
                    let exp = &self.expected[&hash];

                    let mut expected_floats = Vec::new();

                    let step_millis = self.step.as_millis() as u64; // do we have negative steps ?????
                    for (i, e) in exp.vals.iter().enumerate() {
                        let ts = self.start + Duration::from_millis(i as u64 * step_millis);

                        if ts > self.end {
                            let msg = format!("expected {} points for {}, but query time range cannot return this many points",
                                              exp.vals.len(), self.metrics[&hash]);
                            return Err(TestAssertionError::new(self.line, msg));
                        }

                        let timestamp = ts.duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;

                        if !e.omitted {
                            expected_floats.push(Sample {
                                timestamp,
                                value: e.value,
                                metric: s.metric_name.clone(),
                            });
                        }
                    }

                    if expected_floats.len() != s.values.len() {
                        let msg = format!(
                            "expected {} float points for {}, but got {}",
                            expected_floats.len(),
                            self.metrics[&hash],
                            format_series_result(s)
                        );
                        return Err(TestAssertionError::new(self.line, msg));
                    }

                    for (i, expected) in expected_floats.iter().enumerate() {
                        let timestamp = &s.timestamps[i];
                        let value = &s.values[i];

                        if expected.timestamp != *timestamp {
                            let msg = format!("expected float value at index {i} for {} to have timestamp {}, but it had timestamp {} (result has {})",
                                              self.metrics[&hash], expected.timestamp, timestamp, format_series_result(s));
                            return Err(TestAssertionError::new(self.line, msg));
                        }

                        if !almost_equal(*value, expected.value, DEFAULT_EPSILON) {
                            let msg = format!("expected float value at index {} (t={}) for {} to be {}, but got {} (result has {})",
                                              i, timestamp, self.metrics[&hash], expected.value, value, format_series_result(s));
                            return Err(TestAssertionError::new(self.line, msg));
                        }
                    }
                }

                for hash in self.expected.keys() {
                    if !seen.contains_key(hash) {
                        let msg = format!("expected metric {} not found", self.metrics[hash]);
                        return Err(TestAssertionError::new(self.line, msg));
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub(super) fn check_expected_failure(
        &self,
        actual: &RuntimeError,
    ) -> Result<(), TestAssertionError> {
        let error_msg = actual.to_string();
        if let Some(expected) = &self.expected_fail_message {
            if *expected != error_msg {
                let msg = format!("expected error \"{expected}\" evaluating query {} (line {}), but got: {error_msg}",
                                            self.expr, self.line);
                return Err(TestAssertionError::new(self.line, msg));
            }
        }

        if let Some(regex) = &self.expected_fail_regexp {
            if !regex.is_match(&error_msg) {
                let msg = format!("expected error matching pattern {regex} evaluating query {} (line {}), but got: {error_msg}",
                                            self.expr, self.line);
                return Err(TestAssertionError::new(self.line, msg));
            }
        }

        if let Some(cmds) = self.expected_cmds.get(&ExpectCmdType::Fail) {
            if let Some(cmd) = cmds.first() {
                if !cmd.matches(&error_msg) {
                    let msg = format!("expected error matching pattern {cmd} evaluating query {} (line {}), but got: {error_msg}",
                                      self.expr, self.line);
                    return Err(TestAssertionError::new(self.line, msg));
                }
            }
        }

        // We're not expecting a particular error, or we got the error we expected.
        // This test passes.
        Ok(())
    }

    pub(super) fn check_annotations(&self, expr: &str, annos: &Annotations) -> Result<(), TestAssertionError> {
        let (count_warnings, count_info) = annos.count_warnings_and_infos();
        if self.warn && count_warnings == 0 {
            return Err(TestAssertionError::new(
                self.line,
                format!("expected at least one warning for query {expr} (line {})", self.line),
            ));
        }
        if self.info && count_info == 0 {
            return Err(TestAssertionError::new(
                self.line,
                format!("expected at least one info message for query {expr} (line {})", self.line),
            ));
        }
        
        let mut warnings = Vec::new();
        let mut infos = Vec::new();
        for (_, err) in annos.iter() {
            if is_warning(err) {
                warnings.push(err.to_string());
            } else if is_info(err) {
                infos.push(err.to_string());
            }
            else if !err.is_empty() {
                return Err(TestAssertionError::new(
                    self.line,
                    format!("unexpected annotation type, must be either enfo or warn, but got {err} (line {})", self.line),
                ));
            }
        }
        
        if self.warn {
            validate_expected_annotations_of_type(
                &self.expr,
                &self.expected_cmds.get(&ExpectCmdType::Warn).unwrap_or(&Vec::new()),
                &warnings,
                self.line,
                "warning",
                annos,
            )?;
        }
        
        if self.info {
            validate_expected_annotations_of_type(
                &self.expr,
                &self.expected_cmds.get(&ExpectCmdType::Info).unwrap_or(&Vec::new()),
                &infos,
                self.line,
                "info",
                annos,
            )?;
        }
        
        if !warnings.is_empty() {
            match self.expected_cmds.get(&ExpectCmdType::NoWarn) {
                Some(expected) if !expected.is_empty() => {
                    return Err(TestAssertionError::new(
                        self.line,
                        format!("unexpected warnings evaluating query {} (line {}): {:?}", 
                                self.expr, self.line, warnings),
                    ));
                }
                _ => {}
            }
        }
        if !infos.is_empty() {
            match self.expected_cmds.get(&ExpectCmdType::NoInfo) {
                Some(expected) if !expected.is_empty() => {
                    return Err(TestAssertionError::new(
                        self.line,
                        format!("unexpected info messages evaluating query {} (line {}): {:?}", 
                                self.expr, self.line, infos),
                    ));
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// validate_expected_annotations_of_type validates expected messages and regex match actual annotations.
fn validate_expected_annotations_of_type(expr: &str,
                                         expected_annotations: &[ExpectCmd],
                                         actual_annotations: &[String],
                                         line: usize,
                                         annotation_type: &str,
                                         all_annos: &Annotations) -> Result<(), TestAssertionError> {
    if expected_annotations.is_empty() {
        return Ok(())
    }

    if actual_annotations.is_empty() {
        let msg = format!("expected {annotation_type} annotations but none were found for query {expr} (line {line}), expected: {:?}, found: {:?}",
                               expected_annotations, all_annos);
        return Err(TestAssertionError::new(line, msg));
    }

    // Check if all expected annotations are found in actual.
    for e in expected_annotations {
        let match_found = actual_annotations.iter().any(|s| e.matches(s));
        if !match_found {
            let msg = format!("expected {annotation_type} annotation matching {} {e} but no matching annotation was found for query {expr} (line {line}), found: {:?}",
                               e.type_name(), all_annos);
            return Err(TestAssertionError::new(line, msg));
        }
    }

    // Check if all actual annotations have a corresponding expected annotation.
    for anno in actual_annotations {
        let match_found = expected_annotations.iter().any(|e| e.matches(anno));
        if !match_found {
            let msg = format!("unexpected {annotation_type} annotation {anno} found for query {expr} (line {line}), expected: {:?}, found: {:?}",
                               expected_annotations, all_annos);
            return Err(TestAssertionError::new(line, msg));
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub enum TestCommand {
    Clear(ClearCmd),
    Eval(EvalCmd),
    Load(LoadCmd),
}
