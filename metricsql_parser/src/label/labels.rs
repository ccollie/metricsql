// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Label matchers and Well-known label names used by Prometheus components.

use std::collections::BTreeSet;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use ahash::AHashSet;
use serde::{Deserialize, Serialize};

/// "__name__"
pub const METRIC_NAME: &str = "__name__";
/// "alertname"
pub const ALERT_NAME: &str = "alertname";
/// "le"
pub const BUCKET_LABEL: &str = "le";
/// "instance"
pub const INSTANCE_NAME: &str = "instance";

pub type Label = String;

#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq)]
pub struct Labels(pub(crate) Vec<Label>);

impl Labels {
    pub fn append(mut self, l: Label) -> Self {
        self.0.push(l);
        self
    }

    pub fn new(ls: Vec<&str>) -> Self {
        let mut labels: Vec<Label> = ls.iter().map(|s| s.to_string()).collect();
        labels.sort();
        Self(labels)
    }

    pub fn new_from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Label>,
    {
        let mut labels: Vec<Label> = iter.into_iter().collect();
        labels.sort();
        Self(labels)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_joint(&self, ls: &Labels) -> bool {
        let s1: AHashSet<&String> = self.0.iter().collect();
        let s2: AHashSet<&String> = ls.0.iter().collect();

        !s1.is_disjoint(&s2)
    }

    pub fn intersect(&self, ls: &Labels) -> Labels {
        let s1: AHashSet<&String> = self.0.iter().collect();
        let s2: AHashSet<&String> = ls.0.iter().collect();
        let labels = s1.intersection(&s2).map(|s| s.to_string()).collect();

        Self(labels)
    }

    pub fn push_str(&mut self, l: &str) {
        self.0.push(l.to_string());
    }

    pub fn push(&mut self, l: Label) {
        self.0.push(l);
    }

    pub fn remove(&mut self, l: &str) {
        self.0.retain(|x| x != l);
    }

    pub fn sort(&mut self) {
        self.0.sort();
    }

    pub fn iter(&self) -> std::slice::Iter<String> {
        self.0.iter()
    }
}

impl PartialEq<Labels> for Labels {
    fn eq(&self, other: &Labels) -> bool {
        let len = self.0.len();
        if len != other.0.len() {
            return false;
        }
        return match len {
            0 => true,
            1 => self.0[0] == other.0[0],
            _ => {
                let h1: AHashSet<&String> = self.0.iter().collect();
                // compare unsorted
                for label in &other.0 {
                    if !h1.contains(label) {
                        return false;
                    }
                }
                true
            }
        };
    }
}

impl PartialEq<Vec<String>> for Labels {
    fn eq(&self, other: &Vec<String>) -> bool {
        let len = self.0.len();
        if len != other.len() {
            return false;
        }
        return match len {
            0 => true,
            1 => self.0[0] == other[0],
            _ => {
                let h1: AHashSet<&String> = self.0.iter().collect();
                // compare unsorted
                for label in other {
                    if !h1.contains(label) {
                        return false;
                    }
                }
                true
            }
        };
    }
}

impl Hash for Labels {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let len = self.0.len();
        match len {
            0 => (),
            1 => self.0[0].hash(state),
            _ => {
                let sorted = BTreeSet::from_iter(self.0.iter());
                sorted.hash(state);
            }
        }
    }
}

impl FromStr for Labels {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut labels: Vec<Label> = s.split(',').map(|s| s.to_string()).collect();
        labels.sort();
        Ok(Self(labels))
    }
}

impl From<Vec<String>> for Labels {
    fn from(ls: Vec<String>) -> Self {
        let mut labels = ls;
        labels.sort();
        Self(labels)
    }
}

impl AsRef<[String]> for Labels {
    fn as_ref(&self) -> &[String] {
        self.0.as_slice()
    }
}

impl fmt::Display for Labels {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0.join(", "))
    }
}
