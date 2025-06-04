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
use crate::types::MetricName;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use ahash::HashMap;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Sample {
    pub metric: MetricName,
    pub timestamp: i64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ExpectCmdType {
    Ordered,
    Fail,
    Warn,
    NoWarn,
    Info,
    NoInfo
}

impl ExpectCmdType {
    pub fn name(&self) -> &'static str {
        match self {
            ExpectCmdType::Ordered => "ordered",
            ExpectCmdType::Fail => "fail",
            ExpectCmdType::Warn => "warn",
            ExpectCmdType::NoWarn => "no_warn",
            ExpectCmdType::Info => "info",
            ExpectCmdType::NoInfo => "no_info",
        }
    }
}

impl TryFrom<&str> for ExpectCmdType {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let val = hashify::tiny_map_ignore_case! {
            value.to_bytes(),
            "ordered" => ExpectCmdType::Ordered,
            "fail" => ExpectCmdType::Fail,
            "warn" => ExpectCmdType::Warn,
            "no_warn" => ExpectCmdType::NoWarn,
            "info" => ExpectCmdType::Info,
            "no_info" => ExpectCmdType::NoInfo,
        };
        if let Some(cmd_type) = val {
            return Ok(cmd_type);
        }

        Err(format!("unknown expected command type: {value}"))
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct ExpectCmd {
    pub message: String,
    pub regex: Option<Regex>,
}

impl ExpectCmd {
    pub fn new(message: String, regex: Option<Regex>) -> Self {
        Self { message, regex }
    }

    pub fn matches(&self, msg: &str) -> bool {
        if let Some(ref re) = self.regex {
            re.is_match(msg)
        } else {
            self.message == msg
        }
    }

    pub fn type_name(&self) -> &'static str {
        if self.regex.is_some() {
            "regex"
        } else {
            "message"
        }
    }
}

impl Display for ExpectCmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref re) = self.regex {
            write!(f, "expect: {} (regex: {})", self.message, re)
        } else {
            write!(f, "expect: {}", self.message)
        }
    }
}

// SequenceValue struct
#[derive(Debug, Clone)]
pub(crate) struct SequenceValue {
    pub(crate) value: f64,
    pub(crate) omitted: bool,
}

pub enum SequenceValueEnum {
    Stale,
    Missing,
    Value(f64),
    Scalar(SequenceValue),
    Vector(Vec<SequenceValue>),
}

#[derive(Debug, Clone, Default)]
pub struct SeriesDescription {
    pub metric_name: MetricName,
    pub values: Vec<SequenceValue>,
}

pub struct Entry {
    pub pos: usize,
    pub vals: Vec<SequenceValue>
}

#[derive(Debug)]
pub struct ParseErr {
    pub line_offset: usize,
    pub position_range: (usize, usize),
    pub query: String,
    pub err: String,
}

impl Display for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if !self.err.is_empty() {
            write!(f, "Parse error at line {}: {}", self.line_offset, self.err)
        } else {
            write!(
                f,
                "Parse error at line {}: {}",
                self.line_offset, self.query
            )
        }
    }
}

impl Error for ParseErr {}

pub(super) fn raise(line: usize, msg: String) -> ParseErr {
    ParseErr {
        line_offset: line,
        err: msg,
        position_range: (0, 0),
        query: "".to_string(),
    }
}

#[derive(Debug)]
pub struct TestAssertionError {
    pub line_offset: usize,
    pub message: String,
}

impl TestAssertionError {
    pub fn new(line_offset: usize, err: String) -> Self {
        TestAssertionError {
            line_offset,
            message: err,
        }
    }
}

impl Error for TestAssertionError {}

impl Display for TestAssertionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Assertion error at line {}: {}",
            self.line_offset, self.message
        )
    }
}

pub static PROMQL_INFO: &str = "PromQL info";
pub static PROMQL_WARNING: &str = "PromQL warning";

pub(super) fn is_warning(msg: &str) -> bool {
    msg.contains(PROMQL_WARNING)
}

pub(super) fn is_info(msg: &str) -> bool {
    msg.contains(PROMQL_INFO)
}

/// Annotations is a general wrapper for warnings and other information
/// that is returned by the query API along with the results.
/// Each annotation is modeled by a Go error.
/// They are deduplicated based on the string returned by error.Error().
/// The zero value is usable without further initialization, see New().
#[derive(Debug, Clone, Default)]
pub struct Annotations(pub HashMap<String, String>);

impl Annotations {
    pub fn new() -> Self {
        Annotations(HashMap::default())
    }

    pub fn add(&mut self, key: String, value: String) {
        self.0.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    
    pub fn count_warnings(&self) -> usize {
        self.0.iter().filter(|(_, v)| v.contains(PROMQL_WARNING)).count()
    }
    
    pub fn count_infos(&self) -> usize {
        self.0.iter().filter(|(_, v)| v.contains(PROMQL_INFO)).count()
    }
    
    pub fn count_warnings_and_infos(&self) -> (usize, usize) {
        (self.count_warnings(), self.count_infos())
    }
    
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.0.iter()
    }
}