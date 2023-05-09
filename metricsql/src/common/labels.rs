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

use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3;


/// Well-known label names used by Prometheus components.
const METRIC_NAME_LABEL: String = "__name__";
const LABEL_SEP: u8 = 0xfe;
const SEP: u8 = 0xff;

/// Label is a key/value pair of strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label<'a> {
    name: &'a str,
    value: &'a str
}

impl<'a> PartialOrd for Label<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.key == other.key {
            return Some(self.value.cmp(&other.value));
        }
        Some(self.key.cmp(&other.key))
    }
}

// Labels is a sorted set of labels. Order has to be guaranteed upon
// instantiation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Labels(Vec<Label>);

impl Labels {
    pub fn new() -> Labels {
        Labels(Vec::new())
    }

    /// from_map returns new sorted Labels from the given map.
    pub fn from_map(m: HashMap<String, String>) -> Self {
        let mut labels = Vec::with_capacity(m.len());
        for (k, v) in m {
            labels.push(Label{ name: k, value: v });
        }
        labels.sort_by(|a, b| a.name.cmp(b.name));
        Labels(labels)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// has returns true if the label with the given name is present.
    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|l| l.name == name);
    }

    /// get returns the value for the label with the given name.
    /// Returns an empty string if the label doesn't exist.
    fn get(&self, name: &str) -> Option<&String> {
        self.0.iter()
            .find(|label| label.name == name)
            .map(|label| label.value)
    }
    
    pub fn sort(&mut self) -> Self {
        self.0.sort_by(|a, b| {
            let res = a.name.cmp(b.name);
            if res == Ordering::Equal {
                a.value.cmp(b.value)
            } else {
                res
            }
        });
        self
    }

    // Bytes returns itself as a byte slice.
    // It uses an byte invalid character as a separator and so should not be used for printing.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(LABEL_SEP);
        for elem in &self.0[0..] {
            if i > 0 {
                b.push(SEP);
            }
            b.extend_from_slice(elem.name.as_bytes());
            b.push(SEP);
            b.extend_from_slice(elem.value.as_bytes());
        }
        b
    }

    /// match_labels returns a subset of Labels that matches/does not match with the provided label names based on the 'on' boolean.
    /// If on is set to true, it returns the subset of labels that match with the provided label names and its inverse when 'on' is set to false.
    pub fn match_labels(&mut self, on: bool, names: &[String]) -> Labels {
        let matched_labels: Vec<Label> = Vec::with_capacity(self.0.len());

        let name_set = BtreeSet::from_iter(names.iter());
        for v in self.0.iter() {
            let ok = name_set.get(v.name).is_some();
            if on == ok && (on || v.name != METRIC_NAME_LABEL) {
                matched_labels.push(v);
            }
        }

        return matched_labels
    }

    /// hash returns a hash value for the label set.
    /// Note: the result is not guaranteed to be consistent across different runs.
    pub fn hash(&self) -> u64 {
        // Use xxh3_64 for fast path as it's faster.
        let mut b = vec![u8, 1024]; // todo: ArrayVec
        let mut pos = 0;

        for (i, v) in ls.iter().enumerate() {
            if b.len() + v.name.len() + v.value.len() + 2 >= b.capacity() {
                // If labels entry is 1KB+ do not allocate whole entry.
                let mut h = Xxh3::new();
                h.update(b);
                for v in &ls[i..] {
                    push_label(b,h, v);
                }
                return h.digest();
            }

            b.push(v.name.as_bytes());
            b.push(SEP);
            b.push(v.value.as_bytes());
            b.push(SEP)
        }
        xxh3_64(b)
    }

    /// hash_for_labels returns a hash value for the labels matching the provided names.
    /// 'names' have to be sorted in ascending order.
    pub fn hash_for_labels(&self, b: &mut Vec<u8>, names: &[string]) -> u64 {
        b.clear();

        let mut ls_iter = self.0.iter();
        let mut names_iter = names.iter();

        let mut label_ = ls_iter.next();
        let mut name_ = names_iter.next();
        loop {
            match (label_, name_) {
                (Some(label), Some(name)) => {
                    if name < label.name {
                        name_ = names_iter.next(); 
                    } else if label.name < name {
                        label_ = ls_iter.next();
                    } else {
                        push_label(b, label.name, label.value);
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
        let mut j = 0;
        for lbl in self.0.iter() {
            while j < names.len() && names[j] < lbl.name {
                j += 1;
            }
            if lbl.name == METRIC_NAME_LABEL || (j < names.len() && lbl.name == names[j]) {
                continue
            }
            push_label(b, lbl.name, lbl.value);
        }

        xxh3_64(&b)
    }

    /// without_labels returns the labelset without empty labels.
    /// May return the same labelset.
    pub fn without_labels(&self, names: &[String]) -> Labels {
        let mut b = Vec::new();
        self.hash_without_labels(&mut b, names)
    }

    /// Map returns a string map of the labels.
    pub fn as_map(&mut self) -> HashMap<String, String>  {
        let m = HashMap::with_capacity(ls.len());
        for label in self.0.iter() {
            m.insert(l.name, l.value);
        }
        m
    }

    pub fn without_empty(&self) -> Labels {
        let mut els = Vec::with_capacity(self.0.len());
        for label in self.0.iter() {
            if label.value != "" {
                els.push(label);
            }
        }
        els
    }

    /// has_duplicate_label_names returns whether this instance has duplicate label names.
    /// It assumes that the labelset is sorted.
    pub fn has_duplicate_label_names(&self) -> (&str, bool) {
        let prev = self.0[0];
        for i in 1..self.0.len() {
            let curr = self.0[i];
            if curr.name == prev.name {
                return (&curr.name, true)
            }
            prev = curr;
        }
        return ("", false)
    }

    /// validate calls f on each label. If f returns a non-nil error, then it returns that error
    /// cancelling the iteration.
    pub fn validate(&self, f: fn(l: Label) -> Result<()>) -> Result<()> {
        for l in self.0.iter() {
            f(l)?;
        }
        Ok(())
    }
}

fn push_label(b: &mut Vec<u8>, name: &str, value: &str) {
    if b.len() > 1 {
        b.push(SEP)
    }
    b.extend_with_slice(&name);
    b.push(SEP);
    b.extend_with_slice(&value.as_bytes());
}

// isValid checks if the metric name or label names are valid.
fn is_valid() -> bool {
    for l in ls {
        if l.name == METRIC_NAME_LABEL && !model.IsValidMetricname(model.LabelValue(l.value)) {
            return false
        }
        if !model.Labelname(l.name).IsValid() || !model.LabelValue(l.value).IsValid() {
            return false
        }
    }
    return true
}

// Compare compares the two label sets.
// The result will be 0 if a==b, <0 if a < b, and >0 if a > b.
pub fn compare_labels(a: Labels, b: Labels) -> Ordering {
    for (a, b) in a.iter().zip(b.iter()) {
        if a.name != b.name {
            if a.name < b.name {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
        if a.value != b.value {
            if a.value < b.value {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
    }
}


// Builder allows modifying Labels.
pub struct Builder {
    base: Labels,
    del:  Vec<String>,
    add:  Vec<Label>
}

impl Builder {
    pub fn new(base: Labels) -> Self {
        Self {
            base,
            del: Vec::new(),
            add: Vec::new(),
        }
    }

    /// Reset clears all current state for the builder.
    pub fn reset(&mut self, base: Labels) {
        self.base = base;
        self.del.clear();
        self.add.clear();
        for l in &self.base {
            if l.value.is_empty() {
                self.del.push(l.name.clone());
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
            self.del.push(n.clone());
        }
        self
    }
    
    /// Keep removes all labels from the base except those with the given names.
    /// The names are not sorted.
    /// The names are not de-duplicated.
    pub fn keep(&mut self, ns: &[String]) -> &mut Self {
        for l in &self.base {
            let mut found = false;
            for n in ns {
                if l.name == *n {
                    found = true;
                    break;
                }
            }
            if !found {
                self.del.push(l.name.clone());
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
    pub fn get_labels(self) -> Labels {
        let mut res = self.base;
        for n in self.del {
            for i in 0..res.len() {
                if res[i].name == n {
                    res.remove(i);
                    break;
                }
            }
        }
        for add in self.add {
            res.push(add);
        }
        res.sort();
        res
    }
}