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
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub type Sample = crate::provider::Sample;

// SequenceValue struct
#[derive(Debug)]
pub(crate) struct SequenceValue {
    pub(crate) value: f64,
    pub(crate) omitted: bool,
}

#[derive(Debug)]
pub struct ParseErr {
    pub line_offset: usize,
    pub position_range: (usize, usize),
    pub query: String,
    pub err: String
}

impl Display for ParseErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if !self.err.is_empty() {
            write!(f, "Parse error at line {}: {}", self.line_offset, self.err)
        } else {
            write!(f, "Parse error at line {}: {}", self.line_offset, self.query)
        }
    }
}

impl Error for ParseErr {}

pub(super) fn raise(line: usize, msg: String) -> ParseErr {
    ParseErr {
        line_offset: line,
        err: msg,
        position_range: (0, 0),
        query: "".to_string()
    }
}
