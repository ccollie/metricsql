// Copyright 2017 The Prometheus Authors
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

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::vec;
use xxhash_rust::xxh3::xxh3_64;

use crate::parser::ParseResult;

/// Well-known label names used by Prometheus components.
const METRIC_NAME_LABEL: &str = "__name__";
const LABEL_SEP: u8 = 0xfe;
const SEP: u8 = 0xff;

/// Label is a key/value pair of strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord)]
pub struct Label {
    name: String,
    value: String,
}

impl Label {
    fn get_bytes(&self, dst: &mut Vec<u8>) {
        if dst.len() > 1 {
            dst.push(SEP)
        }
        dst.extend_from_slice(self.name.as_bytes());
        dst.push(SEP);
        dst.extend_from_slice(self.value.as_bytes());
    }
}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.name == other.name {
            return Some(self.value.cmp(&other.value));
        }
        Some(self.name.cmp(&other.name))
    }
}

// Labels is a sorted set of labels. Order has to be guaranteed upon
// instantiation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Labels(Vec<Label>);

impl Labels {
    pub fn new(labels: Vec<Label>) -> Labels {
        Labels(labels)
    }

    /// from_map returns new sorted Labels from the given map.
    pub fn from_map(m: HashMap<String, String>) -> Self {
        let mut labels = Vec::with_capacity(m.len());
        for (k, v) in m {
            labels.push(Label { name: k, value: v });
        }
        labels.sort_by(|a, b| a.name.cmp(&b.name));
        Labels(labels)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// has returns true if the label with the given name is present.
    pub fn has(&self, name: &str) -> bool {
        self.0.iter().any(|l| l.name == name)
    }

    /// get returns the value for the label with the given name.
    /// Returns an empty string if the label doesn't exist.
    pub fn get(&self, name: &str) -> Option<&String> {
        self.0
            .iter()
            .find(|label| label.name == name)
            .map(|label| &label.value)
    }

    pub fn add(&mut self, name: String, value: String) {
        self.0.push(Label { name, value });
    }

    pub fn sort(&mut self) {
        self.0.sort();
    }

    // as_bytes returns itself as a byte slice.
    // It uses an byte invalid character as a separator and so should not be used for printing.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(LABEL_SEP);
        for elem in &self.0[0..] {
            elem.get_bytes(&mut b);
        }

        b
    }

    /// match_labels returns a subset of Labels that matches/does not match with the provided label names based on the 'on' boolean.
    /// If on is set to true, it returns the subset of labels that match with the provided label names and its inverse when 'on' is set to false.
    pub fn match_labels(&mut self, on: bool, names: &[String]) -> Labels {
        let mut matched_labels: Vec<Label> = Vec::with_capacity(self.0.len());

        let name_set: HashSet<&String> = HashSet::from_iter(names.iter());
        for v in self.0.iter() {
            let ok = name_set.get(&v.name).is_some();
            if on == ok && (on || v.name != METRIC_NAME_LABEL) {
                matched_labels.push(v.clone());
            }
        }

        Labels(matched_labels)
    }

    /// hash returns a hash value for the label set.
    // pub fn hash(&self) -> u64 {
    //     // Use xxh3_64 for fast path as it's faster.
    //     let mut b = tiny_vec!([u8; 512]);
    //
    //     for (i, v) in self.0.iter().enumerate() {
    //         if b.len() + v.name.len() + v.value.len() + 2 >= b.capacity() {
    //             // If labels entry is 1KB+ do not allocate whole entry.
    //             let mut h = Xxh3::new();
    //             h.update(&b);
    //             for label in &self.0[i..] {
    //                 h.update(&[SEP]);
    //                 h.update(label.name.as_bytes());
    //                 h.update(&[SEP]);
    //                 h.update(label.value.as_bytes());
    //             }
    //             return h.digest();
    //         }
    //         push_label(&mut b, v);
    //     }
    //     xxh3_64(&b)
    // }

    /// hash_for_labels returns a hash value for the labels matching the provided names.
    /// 'names' have to be sorted in ascending order.
    pub fn hash_for_labels(&self, b: &mut Vec<u8>, names: &[String]) -> u64 {
        b.clear();

        let mut ls_iter = self.0.iter();
        let mut names_iter = names.iter();

        let mut label_ = ls_iter.next();
        let mut name_ = names_iter.next();
        loop {
            match (label_, name_) {
                (Some(label), Some(name)) => {
                    if name < &label.name {
                        name_ = names_iter.next();
                    } else if label.name < *name {
                        label_ = ls_iter.next();
                    } else {
                        push_label(b, label);
                        name_ = names_iter.next();
                        label_ = ls_iter.next();
                    }
                }
                (Some(_), None) => break,
                (None, Some(_)) => continue,
                (None, None) => break,
            }
        }

        xxh3_64(&b)
    }

    /// hash_without_labels returns a hash value for all labels except those matching
    /// the provided names.
    /// 'names' have to be sorted in ascending order.
    pub fn hash_without_labels(&self, b: &mut Vec<u8>, names: &[String]) -> u64 {
        b.clear();
        let mut names_iter = names.iter();

        for lbl in self.0.iter() {
            let _ = names_iter.clone().skip_while(|name| *name < &lbl.name);
            let current = names_iter.next();

            if lbl.name == METRIC_NAME_LABEL {
                continue;
            }

            if let Some(name) = current {
                if lbl.name.cmp(name) == Ordering::Equal {
                    continue;
                }
            }

            push_label(b, lbl);
        }

        xxh3_64(&b)
    }

    /// without_labels returns the labelset without empty labels.
    /// May return the same labelset.
    /// 'names' have to be sorted in ascending order.
    pub fn without_labels(&self, names: &[String]) -> Labels {
        let mut b = Vec::new();
        let mut names_iter = names.iter();
        for lbl in self.0.iter() {
            let _ = names_iter.clone().skip_while(|name| *name < &lbl.name);
            let current = names_iter.next();
            if lbl.name == METRIC_NAME_LABEL {
                continue;
            }
            if let Some(name) = current {
                if lbl.name.cmp(name) == Ordering::Equal {
                    continue;
                }
            }
            b.push(Label {
                name: lbl.name.clone(),
                value: lbl.value.clone(),
            });
        }

        Labels(b)
    }

    /// Map returns a string map of the labels.
    pub fn as_map(&mut self) -> HashMap<&String, String> {
        let mut m = HashMap::with_capacity(self.len());
        for label in self.0.iter() {
            m.insert(&label.name, label.value.clone());
        }
        m
    }

    pub fn without_empty(&self) -> Labels {
        let mut els = Vec::with_capacity(self.0.len());
        for label in self.0.iter() {
            if label.value != "" {
                els.push(label.clone());
            }
        }
        Labels(els)
    }

    /// has_duplicate_label_names returns whether this instance has duplicate label names.
    /// It assumes that the labelset is sorted.
    pub fn has_duplicate_label_names(&self) -> (&str, bool) {
        let mut prev = &self.0[0];
        for curr in self.0.iter().skip(1) {
            if curr.name == prev.name {
                return (&curr.name, true);
            }
            prev = curr;
        }
        return ("", false);
    }

    /// validate calls f on each label. If f returns a non-nil error, then it returns that error
    /// cancelling the iteration.
    pub fn validate(&self, f: fn(l: &Label) -> ParseResult<()>) -> ParseResult<()> {
        for l in self.0.iter() {
            f(l)?;
        }
        Ok(())
    }
}

impl Default for Labels {
    fn default() -> Self {
        Labels(Vec::new())
    }
}

fn push_label(b: &mut Vec<u8>, label: &Label) {
    label.get_bytes(b);
}

// Compare compares the two label sets.
// The result will be 0 if a==b, <0 if a < b, and >0 if a > b.
pub fn compare_labels(a: Labels, b: Labels) -> Ordering {
    for (a, b) in a.0.iter().zip(b.0.iter()) {
        let mut cmp = a.name.cmp(&b.name);
        if cmp == Ordering::Equal {
            cmp = a.value.cmp(&b.value);
        }
        if cmp != Ordering::Equal {
            return cmp;
        }
    }
    Ordering::Equal
}

// Builder allows modifying Labels.
pub struct Builder {
    base: Labels,
    del: HashSet<String>,
    add: Vec<Label>,
}

impl Builder {
    pub fn new(base: Labels) -> Self {
        Self {
            base,
            del: HashSet::with_capacity(16),
            add: vec![],
        }
    }

    /// Reset clears all current state for the builder.
    pub fn reset(&mut self, base: Labels) {
        self.base = base;
        self.del.clear();
        self.add.clear();
        for l in self.base.0.iter() {
            if l.value.is_empty() {
                self.del.insert(l.name.clone());
            }
        }
    }

    /// del deletes the label of the given name.
    pub fn del(&mut self, ns: &[String]) -> &mut Self {
        for n in ns {
            for i in 0..self.add.len() {
                if self.add[i].name == *n {
                    self.add.remove(i);
                }
            }
            self.del.insert(n.clone());
        }
        self
    }

    /// Keep removes all labels from the base except those with the given names.
    /// The names are not sorted.
    /// The names are not de-duplicated.
    pub fn keep(&mut self, ns: &[String]) -> &mut Self {
        for l in self.base.0.iter() {
            let mut found = false;
            for n in ns {
                if l.name == *n {
                    found = true;
                    break;
                }
            }
            if !found {
                self.del.insert(l.name.clone());
            }
        }
        self
    }

    /// Set the name/value pair as a label. A value of "" means delete that label.
    pub fn set(&mut self, n: String, v: String) -> &mut Self {
        if v.is_empty() {
            // Empty labels are the same as missing labels.
            return self.del(&[n]);
        }
        for add in self.add.iter_mut() {
            if add.name == n {
                add.value = v;
                return self;
            }
        }
        self.add.push(Label { name: n, value: v });
        self
    }

    /// Labels returns the labels from the builder, adding them to res if non-nil.
    /// Argument res can be the same as b.base, if caller wants to overwrite that slice.
    /// If no modifications were made, the original labels are returned.
    pub fn get_labels(&mut self) -> &Labels {
        self.base.0.retain(|l| !self.del.contains(&l.name));
        for add in self.add.iter() {
            self.base.0.push(add.clone());
        }
        self.base.sort();
        &self.base
    }
}
