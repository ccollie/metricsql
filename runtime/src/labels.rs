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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fmt::Display;
use tinyvec::*;
use metricsql::types::ReturnKind::String;

// https://github.com/bboreham/prometheus/tree/labels-as-string

// Well-known label names used by Prometheus components.
static MetricName: &str = "__name__";
static AlertName: &str = "alertname";
static BucketLabel: &str = "le";
static InstanceName: &str = "instance";

static seps: &str = b'\xff'.to_string();

// Label is a key/value pair of strings.
pub struct Label<'a> {
    pub name: &'a str,
    pub value: &'a str
}

// Labels is implemented by a single flat string holding name/value pairs.
// Each name and value is preceded by its length in varint encoding.
// Names are in order.
#[derive(Clone, Default, Debug, Hash)]
pub struct Labels {
    data: String // tinyvec
}

impl Labels {

    // New returns a sorted Labels from the given labels.
    // The caller has to guarantee that all label names are unique.
    pub fn new(ns: &mut [Label]) -> Labels {
        ns.sort_by(|a, b| {
            let cmp = a.name.cmp(b.name);
            if cmp == Ordering::Equal {
                return a.value.cmp(b.value);
            }
            return cmp;
        });
        let size = labels_size(ls);
        let mut buf: Vec<u8> = Vec::with_capacity(size);
        marshal_labels_to_sized_buffer(ls, buf);
        return Labels{data: yoloString(buf)}
    }

    pub fn len(self) -> usize { self.data.len() };

    // IsZero implements yaml.IsZeroer - if we don't have this then 'omitempty' fields are always omitted.
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    // Call f on each label; return a new Labels of those where f returns true.
    pub fn filtered(self, f: fn(l: Label) -> bool) -> Labels {
        let labels = self.labels().filter(f);
        return Labels{data: yoloString(buf)}
    }

    // Call f on each label.
    pub fn range(self, f: fn(l: Label)) {
        for label in self.labels() {
            f(label);
        }
    }

    // MatchLabels returns a subset of Labels that matches/does not match with the provided label names
    // based on the 'on' boolean. If on is set to true, it returns the subset of labels that match with
    // the provided label names and its inverse when 'on' is set to false.
    // TODO: This is only used in printing an error message
    pub fn match_labels(self, on: bool, names: &[String]) -> Labels {
        let mut b = NewBuilder(ls);
        if on {
            b.keep(names);
        } else {
            b.del(MetricName);
            b.del(names);
        }
        return b.Labels()
    }

    // WithoutEmpty returns the labelset without empty labels.
    // May return the same labelset.
    pub fn without_empty(&mut self) -> Labels {
        let mut pos = 0;
        let mut size: usize = 0;
        let mut filtered: tiny_vec!([Label; 10]);
        for label in self.labels() {
            if label.value.len() > 0 {
                // get rough size for alloc
                size += label.value.len() + label.name.len() + 2;
                filtered.push(label);
            }
        }
        let mut buf = String::with_capacity(size);
        for label in filtered {

        }
    }

    // hash_without_labels returns a hash value for all labels except those matching
    // the provided names.
    // 'names' have to be sorted in ascending order.
    pub fn hash_without_labels(self, b: &mut Vec<u8>, names: &[String]) -> u64 {
        let name_set: HashSet<String> = HashSet::from(names);
        for label in self.labels() {
            if name == MetricName || name_set.contains(label.name) {
                continue;
            }
            b.extend_from_slice(label.name.as_bytes());
            b.push(seps[0]);
            b.extend_from_slice(label.value.as_bytes());
            b.push(seps[0]);
        }
        return xxhash.Sum64(b)
    }

    // hash_for_labels returns a hash value for the labels matching the provided names.
    // 'names' have to be sorted in ascending order.
    fn hash_for_labels(self, b: &mut Vec<u8>, names: &[String]) -> u64 {
        let name_set: HashSet<String> = names.iter().cloned().collect();
        for label in self.labels() {
            if name_set.contains(label.name) {
                b.extend_from_slice(label.name.as_bytes());
                b.push(seps[0]);
                b.extend_from_slice(label.value.as_bytes());
                b.push(seps[0]);
            }
        }
        return xxhash.Sum64(b)
    }

    // BytesWithoutLabels is just as Bytes(), but only for labels not matching names.
    // 'names' have to be sorted in ascending order.
    pub fn bytes_without_labels(self, buf: &mut Vec<u8>, names: &[String]) -> &[u8] {
        let name_set: HashSet<String> = names.iter().cloned().collect();
        for label in self.labels() {
            if !name_set.contains(label.name) {
                buf.push_str(label.name.as_bytes());
                buf.push_str(label.value.as_bytes())
            }
        }
        &buf[0..]
    }

    // bytes_with_labels is just as Bytes(), but only for labels matching names.
    // 'names' have to be sorted in ascending order.
    pub fn bytes_with_labels(self, buf: &mut Vec<u8>, names: &[String]) {
        let mut pos = 0;
        let set: HashSet<String> = HashSet::from(names.into());
        while pos < self.data.len()  {
            let (l_name, mut new_pos) = decode_string(ls.data, pos);
            let (_, x) = decode_string(ls.data, new_pos);
            new_pos = x;
            if set.contains(l_name) {
                buf.extend_from_slice(self.data[pos ..new_pos].as_ref());
            }
            pos = new_pos
        }
        return b
    }

    // Get returns the value for the label with the given name.
    // Returns an empty string if the label doesn't exist.
    pub fn get(name: &str) -> &str {
        let mut i = 0;
        while i < ls.data.len() {
            let (l_name, mut i) = decode_string(ls.data, i);
            let (l_value, x) = decode_string(ls.data, i);
            i = x;
            if l_name == name {
                return l_value
            }
        }
        return ""
    }

    // Map returns a string map of the labels.
    pub fn get_map(self) -> HashMap<&str, &str> {
        let mut m: HashMap<&str, &str> = HashMap::with_capacity(self.len()/10);
        for label in self.labels() {
            m.insert(label.nane, label.value);
        }
        return m
    }

    // Has returns true if the label with the given name is present.
    pub fn has(self, name: &str) -> bool {
        self.labels().any(|l| l.name == name)
    }

    // HasDuplicateLabelNames returns whether ls has duplicate label names.
    // It assumes that the labelset is sorted.
    pub fn has_duplicate_label_names(self) -> (&str, bool) {
        let mut i = 0;
        while i < ls.data.len() {
            let (l_name, x) = decode_string(ls.data, i);
            i = x;
            let (_, x) = decode_string(ls.data, i);
            i = x;
            if l_name == prevName {
                return (l_name, true);
            }
            prevName = l_name
        }
        return ("", false);
    }

    // Merge external_labels into ls. If ls contains
    // a label in external_labels, the value in ls wins.
    fn merge(&mut self, external_labels: &Labels) -> Labels {
        let mut size: usize = 0;

        if external_labels.is_empty() {
            return ls
        } else if self.is_empty() {
            return external_labels
        }
        let mut left: BTreeMap<&str, &Label> = BTreeMap::new();
        for label in self.labels() {
            size += label_size(&label);
            left.insert(&label.name, &label);
        }

        for label in external_labels.labels() {
            if !left.contains_key(label.name) {
                size += label_size(&label);
                left.insert(&label.name, &label);
            }
        }

        let mut buf: Vec<u8> = Vec::with_capacity(size);
        buf.append(append(buf, ls.data[prev:]...), externalLabels.data[ePrev:]...)
        return Labels{data: yoloString(buf)}
    }

    pub fn labels(self) -> LabelIterator {
        LabelIterator {
            labels: &self,
            index: 0
        }
    }
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl PartialOrd for Labels {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.path.partial_cmp(&other.path)
    }
}

impl Display for Labels {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.metric_group)?;
        let len = self._items.len();
        let mut i = 0;
        for label in self.labels() {
            write!(f, "{}={}", label.name, enquote('"', label.value))?;
            if i < len - 1 {
                write!(f, ",")?;
            }
            i = i + 1;
        }
        write!(f, "}}")?;
        Ok(())
    }
}


fn decode_size(data: &str, index: usize) -> (usize, usize) {
    let mut size: usize = 0;
    let mut index = index;
    let mut shift: usize = 0;
    loop {
        // Just panic if we go of the end of data, since all Labels strings are constructed internally and
        // malformed data indicates a bug, or memory corruption.
        let b = &data[index];
        index += 1;
        size |= int(b&0x7F) << shift;
        if b < 0x80 {
            break
        }

        shift += 7;
    }

    return (size, index)
}

fn decode_string(data: &string, index: usize) -> (&str, usize) {
    let (size, index) = decode_size(data, index);
    return (&data[index .. index+size], index + size);
}


// FromMap returns new sorted Labels from the given map.
fn FromMap(m: HashMap<String, String>) -> Labels {
    let l = make([]Label, 0, len(m))
    for k, v := range m {
        l = append(l, Label{Name: k, Value: v})
    }
    return New(l...)
}

// from_strings creates new labels from pairs of strings.
fn from_strings(ss: &[String]) -> Labels {
    if ss.len() % 2 != 0 {
        panic!("invalid number of strings")
    }
    let mut ls: Vec<Label> = Vec::with_capacity(ss.len() / 2);
    let mut i = 0;
    while i < ss.len() {
        ls.push( Label{ name: ss[i].as_str(), value: ss[i+1].as_str() });
        i += 2;
    }

    Labels::new(&ls)
}


fn EmptyLabels() Labels {
return Labels{}
}

// IsEmpty returns true if ls represents an empty set of labels.
fn (ls Labels) IsEmpty() bool {
return self.len() == 0
}

// Len returns the number of labels; it is relatively slow.
fn (ls Labels) Len() int {
    let mut count = 0;
    for i = 0; i < self.len(); {
    var size int
    size, i = decodeSize(ls.data, i)
i += size
size, i = decodeSize(ls.data, i)
i += size
count++
}
return count
}


// Call f on each label; if f returns true then replace the Label
// Name of returned label must have same value as original.
fn (ls Labels) replace(&mut self, f: fn(l: Label) -> (&str, bool)) -> Labels {
    let bufSize = self.data.len()
    let size: uint = 0;
    let mut replacements: HashMap<&str, &str> = HashMap::new();
    let mut items: BtreeMap<&str, &str> = Vec::with_capacity(self.len());

    let mut changed: bool = false;

    for label in self.labels() {
        let (replacement, change) = f(label);
        if change {
            items.insert(label.name, replacement);
            size += label.len() + replacement.len() + 2;
        } else {
            items.insert(label.name, label.value)
            size += label_size(label);
        }
    }

    marshal_labels_to_sized_buffer(&v, buf)
} else if buf != nil {
    buf = append(buf, ls.data[pos:newPos]...)
}
pos = newPos
}
if buf == nil { // Nothing was replaced.
return ls
}
return Labels{data: yoloString(buf)}
}


// Builder allows modifying Labels.
pub struct Builder<'a> {
    base: Labels,
    del: HashSet<String>,
    add: HashMap<&'a str, Label<'a>>
}

impl<'a> Builder<'a> {
    pub fn new() -> Self {
        Self {
            base: Default::default(),
            del: Default::default(),
            add: Default::default()
        }
    }

    // Reset clears all current state for the builder.
    pub fn reset(&mut self, base: Labels) -> Self {
        self.base = base;
        self.del.clear();
        self.add.clear();
        for label in base.labels() {
            if label.value.len() == 0 {
                self.del.push(label.name.to_string());
            }
        }
        self
    }

    // Del deletes the label of the given name.
    pub fn del(&mut self, items: &[String]) {
        // Del deletes the label of the given name.
        for item in items {
            self.add.remove(item);
            self.del.push(item);
        }
    }

    // Keep removes all labels from the base except those with the given names.
    pub fn keep(&mut self, ns: &[String]) -> Self {
        for label in self.base.labels() {
            if ns.contains(&label.name.to_string()) {
                continue
            }
            b.del.push(lName);
        }
        self
    }

    // Set the name/value pair as a label.
    pub fn set(mut self, n: &str, v: &str) -> Self {
        if v.len() == 0 {
            // Empty labels are the same as missing labels.
            return self.del(n.to_string())
        }
        for (i, a) in b.add.iter().enumerate() {
            if a.name == n {
                b.add[i].value = v;
                return self;
            }
        }
        self.add.push(Label{name: n, value: v});

        self
    }


    // Labels returns the labels from the builder. If no modifications
    // were made, the original labels are returned.
    pub fn labels(&mut self) -> Labels {
        if self.del.len() == 0 && self.add.len() == 0 {
            return self.base
        }

        self.add.sort(|a, b| a.label.cmp(b.label));

        let buf: Vec<Label> = Vec::with_capacity(b.base.data.len());
        for label in self.base.labels() {
            if self.del.contains(label.name) {
                continue;
            }
        }
        let mut oldPos = 0;
        let mut pos: usize = 0;
        while pos < self.base.data.len() {
            oldPos = pos;
        }
        let (lName, pos) = decode_string(b.base.data, pos);
        _, pos = decode_string(b.base.data, pos);
        while a < b.add.len() && b.add[a].name < lName {
            buf.push(&b.add[a]); // Insert label that was not in the base set.
            a += 1;
        }
        if a < b.add.len() && b.add[a].Name == lName {
            buf.push(&b.add[a]);
            a += 1;
            continue // This label has been replaced.
        }
        buf.append(b.base.data[oldPos:pos]...)
    }
    // We have come to the end of the base set; add any remaining labels.
    while a < b.add.len() {
        buf = appendLabelTo(buf, &b.add[a])
        a += 1;
    }
    return Labels{data: yoloString(buf)}
    }
}


fn marshal_labels_to_sized_buffer(lbls: &[Label], data: &mut [u8]) -> usize {
    let mut index = data.len() - 1;
    while index >= 0 {
        size = marshal_label_to_sized_buffer(&lbls[index], data[index]);
        index -= size
    }
    data.len() - i
}

fn marshal_label_to_sized_buffer(m: &Label, data: &mut [u8]) -> usize {
    let mut i = data.len();
    i -= m.value.len();
    copy(data[i:], m.value);
    i = encode_size(data, i, m.value.len());
    i -= m.name.len();
    copy(data[i:], m.name)
    i = encode_size(data, i, m.name.len())
    return data.len() - i;
}

fn size_varint(x: u64) -> usize {
    let mut x = x;
    // Most common case first
    if x < 1<<7 {
        return 1
    }
    if x >= 1<<56 {
        return 9
    }
    if x >= 1<<28 {
        x >>= 28;
        n = 4;
    }
    if x >= 1<<14 {
        x >>= 14;
        n += 2
    }
    if x >= 1<<7 {
        n += 1;
    }
    return n + 1;
}

fn encode_varint(data: &mut [u8], offset: usize, v: i64) -> usize {
    let mut ofs = offset;
    ofs -= size_varint(v);
    let base = offset;
    let mut v = v;
    let offset = offset;
    while v >= 1<<7 {
        data[offset] = (v&0x7f | 0x80) as u8;
        v >>= 7;
        ofs += 1;
    }
    data[ofs] = v as u8;
    return base
}

// Special code for the common case that a size is less than 128
fn encode_size(data: &mut [u8], offset: usize, v: usize) -> usize {
    let mut offset = offset;
    if v < 1<<7 {
        offset -= 1;
        data[offset] = v as u8;
        return offset
    }
    return encode_varint(data, offset, uint64(v))
}

fn labels_size(lbls: &[Label]) -> usize {
    // we just encode name/value/name/value, without any extra tags or length bytes
    let mut n: usize = 0;
    for e in lbls {
        n += label_size(e);
    }
    return n
}

fn label_size(m: &Label) -> usize {
    let mut n: usize = 0;
    // strings are encoded as length followed by contents.
    let mut l = m.name.len();
    n += l + size_varint(l as u64);
    l = m.value.len();
    n += l + size_varint(l as u64);
    return n
}

fn append_label_to(buf: &mut Vec<u8>, m: &Label) {
    let mut size = label_size(m);
    buf.reserve(buf.len() + size);
    marshal_labels_to_sized_buffer(m, buf)
}

// SimpleBuilder allows efficient construction of a Labels from scratch.
pub(crate) struct SimpleBuilder {
    add: Vec<Label>
}

impl SimpleBuilder {
    pub fn reset(mut self) {
        self.add.clear()
    }

    // Add a name/value pair.
    // Note if you Add the same name twice you will get a duplicate label, which is invalid.
    pub fn add(mut self, name: &str, value: &str) {
        let label = Label {
            name: name.to_string(),
            value: value.to_string()
        };
        self.add.push(label);
    }

    // Sort the labels added so far by name.
    pub fn sort(mut self) -> Self {
        self.add.sort_by(|a, b| a.name.cmp(b.name));
        self
    }
}

// Return the name/value pairs added so far as a Labels object.
// Note: if you want them sorted, call Sort() first.
func (b *SimpleBuilder) Labels() Labels {
let size = labelsSize(b.add)
buf = make([]byte, size)
marshal_labels_to_sized_buffer(b.add, buf)
return Labels{data: yoloString(buf)}
}

// Write the newly-built Labels out to ls, reusing its buffer if long enough.
// Callers must ensure that there are no other references to ls.
fn (b *SimpleBuilder) Overwrite(ls: &Labels) {
    let size = labelsSize(b.add)
var buf []byte
if size <= self.len() {
buf = yoloBytes(ls.data)[:size]
} else {
buf = make([]byte, size)
}
marshal_labels_to_sized_buffer(b.add, buf)
ls.data = yoloString(buf)
}

/// An iterator over the non-zero buckets in a histogram.
#[derive(Debug, Clone)]
pub struct LabelIterator<'a> {
    labels: &'a Labels,
    index: usize,
}

impl<'a> Iterator for LabelIterator<'a> {
    type Item = Label<'a>;

    fn next(&mut self) -> Option<Self::Item> {

        if self.index >= self.labels.len() {
            return None;
        }

        let (name, x)  = decode_string(&self.labels.data, self.index);
        self.index = x;
        let (value, x) = decode_string(&self.labels.data, self.index);
        self.index = x;


        Some(Label {
            name,
            value
        })
    }
}