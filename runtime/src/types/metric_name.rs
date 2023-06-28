use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use enquote::enquote;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use metricsql::common::{AggregateModifier, GroupModifier, GroupModifierOp};

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::signature::Signature;
use crate::utils::{read_string, read_usize, write_string, write_usize};

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const METRIC_NAME_LABEL: &str = "__name__";

const SEP: u8 = 0xff;

// for tag manipulation (removing, adding, etc), name vectors longer than this will be converted to a hashmap
// for comparison, otherwise we do a linear search
const SET_SEARCH_MIN_THRESHOLD: usize = 8;

/// Tag represents a (key, value) tag for metric.
#[derive(Debug, Default, PartialEq, Eq, Clone, Hash, Ord, Serialize, Deserialize)]
pub struct Tag {
    pub key: String,
    pub value: String,
}

impl Tag {
    pub fn new<S: Into<String>>(key: S, value: String) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }

    pub fn marshal(&self, buf: &mut Vec<u8>) {
        write_string(buf, &self.key);
        write_string(buf, &self.value);
    }

    fn unmarshal<'a>(&mut self, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
        let (src, key) = read_string(src, "tag key")?;
        let (src, value) = read_string(src, "tag value")?;
        self.key = key;
        self.value = value;
        Ok(src)
    }

    pub(crate) fn update_hash(&self, h: &mut Xxh3) {
        h.update(self.key.as_bytes());
        h.write_u8(SEP);
        h.update(self.value.as_bytes());
    }
}

impl PartialOrd for Tag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.key == other.key {
            return Some(self.value.cmp(&other.value));
        }
        Some(self.key.cmp(&other.key))
    }
}

/// MetricName represents a metric name.
#[derive(Debug, PartialEq, Eq, Clone, Default, Hash, Serialize, Deserialize)]
pub struct MetricName {
    pub metric_group: String,
    // todo: Consider https://crates.io/crates/btree-slab or heapless btree to minimize allocations
    pub tags: Vec<Tag>,
    #[serde(skip)]
    pub(crate) hash: Option<u64>,
    #[serde(skip)]
    sorted: bool,
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            metric_group: name.to_string(),
            tags: vec![],
            hash: None,
            sorted: true,
        }
    }

    /// from_strings creates new labels from pairs of strings.
    pub fn from_strings(ss: &[&str]) -> RuntimeResult<Self> {
        if ss.len() % 2 != 0 {
            return Err(RuntimeError::from("invalid number of strings"));
        }

        let mut res = MetricName::default();
        for i in (0..ss.len()).step_by(2) {
            res.set_tag(ss[i], ss[i + 1]);
        }

        Ok(res)
    }

    pub fn reset_metric_group(&mut self) {
        self.metric_group = "".to_string();
    }

    pub fn copy_from(&mut self, other: &MetricName) {
        self.metric_group = other.metric_group.clone();
        self.tags = other.tags.clone();
    }

    /// Reset resets the mn.
    pub fn reset(&mut self) {
        self.metric_group = "".to_string();
        self.tags.clear();
        self.hash = None;
        self.sorted = true;
    }

    pub fn set_metric_group(&mut self, value: &str) {
        self.metric_group = value.to_string();
    }

    /// add_tag adds new tag to mn with the given key and value.
    pub fn add_tag<S: Into<String>>(&mut self, key: &str, value: S) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.into();
            return;
        }
        self.upsert(key, value.into());
        self.hash = None;
    }

    pub fn add_tags_from_hashmap(&mut self, tags: &HashMap<String, String>) {
        for (key, value) in tags {
            if key == METRIC_NAME_LABEL {
                self.metric_group = value.into();
                return;
            }
            self.upsert(key, value.clone());
        }
        self.hash = None;
    }

    fn upsert(&mut self, key: &str, value: String) {
        match self.tags.binary_search_by_key(&key, |tag| &tag.key) {
            Ok(idx) => self.tags[idx].value = value,
            Err(idx) => {
                let tag = Tag {
                    key: key.to_string(),
                    value,
                };
                self.tags.insert(idx, tag);
            }
        }
    }

    /// adds new tag to mn with the given key and value.
    pub fn set_tag<S: Into<String>>(&mut self, key: &str, value: S) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.into();
            self.hash = None;
            return;
        } else {
            self.upsert(key, value.into());
        }
    }

    /// removes a tag with the given tagKey
    pub fn remove_tag(&mut self, key: &str) {
        if key == METRIC_NAME_LABEL {
            self.reset_metric_group();
        } else {
            let count = self.tags.len();
            self.tags.retain(|x| x.key != key);
            if count != self.tags.len() {
                self.hash = None;
            }
        }
    }

    pub fn has_tag(&self, key: &str) -> bool {
        self.tags.iter().position(|x| x.key == key).is_some()
    }

    /// returns tag value for the given tagKey.
    pub fn get_tag_value(&self, key: &str) -> Option<&String> {
        if key == METRIC_NAME_LABEL {
            return Some(&self.metric_group);
        }
        self.tags
            .binary_search_by_key(&key, |tag| &tag.key)
            .ok()
            .map(|index| &self.tags[index].value)
    }

    /// remove_tags_on removes all the tags not included to on_tags.
    pub fn remove_tags_on(&mut self, on_tags: &Vec<String>) {
        let set: HashSet<_> = HashSet::from_iter(on_tags);
        // written this way to avoid an allocation
        // (to compare against METRIC_NAME_LABEL.to_string())
        if !set.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_metric_group()
        }
        if on_tags.len() >= SET_SEARCH_MIN_THRESHOLD {
            let set: HashSet<_> = HashSet::from_iter(on_tags);
            self.tags.retain(|tag| set.contains(&tag.key));
        } else {
            self.tags.retain(|tag| on_tags.contains(&tag.key));
        }
        self.hash = None;
    }

    /// remove_tags_ignoring removes all the tags included in ignoring_tags.
    pub fn remove_tags_ignoring(&mut self, ignoring_tags: &Vec<String>) {
        if ignoring_tags.is_empty() {
            return;
        }
        if ignoring_tags
            .iter()
            .any(|x| x.as_str() == METRIC_NAME_LABEL)
        {
            self.reset_metric_group();
        }

        if ignoring_tags.len() >= SET_SEARCH_MIN_THRESHOLD {
            let set: HashSet<_> = HashSet::from_iter(ignoring_tags);
            self.tags.retain(|tag| !set.contains(&tag.key));
        } else {
            self.tags.retain(|tag| !ignoring_tags.contains(&tag.key));
        }
        self.hash = None;
    }

    pub fn update_tags_by_group_modifier(&mut self, modifier: &GroupModifier) {
        match modifier.op {
            GroupModifierOp::On => {
                self.remove_tags_on(&modifier.labels);
            }
            GroupModifierOp::Ignoring => {
                self.remove_tags(&modifier.labels);
            }
        }
    }

    /// removes all the tags included in ignoring_tags.
    pub fn remove_tags(&mut self, ignoring_tags: &[String]) {
        let mut set: HashSet<&String> = HashSet::with_capacity(ignoring_tags.len());
        for tag in ignoring_tags {
            if tag == METRIC_NAME_LABEL {
                self.reset_metric_group();
            } else {
                set.insert(tag);
            }
        }
        self.tags.retain(|tag| set.contains(&tag.key));
        self.hash = None;
    }

    /// sets tags from src with keys matching add_tags.
    pub(crate) fn set_tags(&mut self, add_tags: &[String], src: &mut MetricName) {
        for tag_name in add_tags {
            if tag_name == METRIC_NAME_LABEL {
                self.metric_group = tag_name.clone();
                continue;
            }

            // todo: use iterators instead
            match src.get_tag_value(tag_name) {
                Some(tag_value) => {
                    self.set_tag(tag_name, tag_value);
                }
                None => {
                    self.remove_tag(tag_name);
                }
            }
        }
        self.hash = None;
    }

    pub fn append_tags_to_string(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice("{{".as_bytes());

        let len = &self.tags.len();
        let mut i = 0;
        for Tag { key: k, value: v } in self.tags.iter() {
            dst.extend_from_slice(format!("{}={}", k, enquote('"', &v)).as_bytes());
            if i + 1 < *len {
                dst.extend_from_slice(", ".as_bytes())
            }
            i += 1;
        }
        // ??????????????
        dst.extend_from_slice('}'.to_string().as_bytes());
    }

    pub(crate) fn marshal_tags_fast(&self, dst: &mut Vec<u8>) {
        // Calculate the required size and pre-allocate space in dst
        let required_size = self
            .tags
            .iter()
            .fold(0, |acc, tag| acc + tag.key.len() + tag.value.len() + 8);
        dst.reserve(required_size + 4);
        write_usize(dst, self.tags.len());
        for tag in self.tags.iter() {
            tag.marshal(dst);
        }
    }

    fn unmarshal_tags<'a>(&mut self, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
        let (mut src, len) = read_usize(src, "tag count")?;
        self.tags = Vec::with_capacity(len);
        for _ in 0..len {
            let mut tag = Tag::default();
            src = tag.unmarshal(src)?;
            self.tags.push(tag);
        }
        Ok(src)
    }

    /// marshal appends marshaled mn to dst.
    ///
    /// `self.sort_tags` must be called before calling this function
    /// in order to sort and de-duplicate tags.
    pub fn marshal(&self, dst: &mut Vec<u8>) {
        // Calculate the required size and pre-allocate space in dst
        let required_size = self.metric_group.len() + 8;
        dst.reserve(required_size);

        write_string(dst, &self.metric_group);
        self.marshal_tags_fast(dst);
    }

    /// unmarshal unmarshals mn from src.
    /// Todo(perf) this is not necessarily as performant as can be, even for
    /// simple serialization
    pub fn unmarshal(src: &[u8]) -> RuntimeResult<(&[u8], MetricName)> {
        let mut mn = MetricName::default();
        let (mut src, group) = read_string(src, "metric group")?;

        mn.metric_group = group;
        src = &mn.unmarshal_tags(src)?;
        Ok((src, mn))
    }

    pub(crate) fn serialized_size(&self) -> usize {
        let mut n = 2 + self.metric_group.len();
        n += 2; // Length of tags.
        for Tag { key: k, value: v } in self.tags.iter() {
            n += 2 + k.len();
            n += 2 + v.len();
        }
        n
    }

    pub fn remove_group_tags(&mut self, modifier: &Option<AggregateModifier>) {
        if let Some(m) = modifier.deref() {
            match m {
                AggregateModifier::By(labels) => {
                    self.remove_tags_on(labels);
                }
                AggregateModifier::Without(labels) => {
                    self.remove_tags(labels);
                    // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                    self.reset_metric_group();
                }
            }
        };
    }

    pub(crate) fn count_label_values(&self, hm: &mut HashMap<String, HashMap<String, usize>>) {
        // duplication, I know
        let label_counts = hm
            .entry(METRIC_NAME_LABEL.to_string())
            .or_insert_with(|| HashMap::new());
        *label_counts
            .entry(self.metric_group.to_string())
            .or_insert(0) += 1;
        for tag in self.tags.iter() {
            count_label_value(hm, &tag.key, &tag.value);
        }
    }

    pub(crate) fn fast_hash(&self, hasher: &mut Xxh3) -> u64 {
        hasher.reset();
        hasher.update(self.metric_group.as_bytes());
        for tag in self.tags.iter() {
            tag.update_hash(hasher);
        }
        hasher.digest()
    }

    pub(crate) fn signature(&self) -> Signature {
        Signature::new(self)
    }

    pub fn sort_tags(&mut self) {
        if !self.sorted {
            self.tags.sort();
            self.sorted = true;
        }
    }

    pub fn with_labels_iter<'a>(&'a self, names: &'a [String]) -> impl Iterator<Item=&Tag> {
        WithLabelsIterator::new(self, names)
    }

    pub fn without_labels_iter<'a>(&'a self, names: &'a [String]) -> impl Iterator<Item=&Tag> {
        WithoutLabelsIterator::new(self, names)
    }

    /// `names` have to be sorted in ascending order.
    pub fn signature_with_labels(&self, names: &[String]) -> Signature {
        Signature::from_tag_iter(self.with_labels_iter(names))
    }

    pub fn signature_without_labels(&self, names: &[String]) -> Signature {
        Signature::from_tag_iter(self.without_labels_iter(names))
    }

    pub fn signature_by_group_modifier(
        &self,
        modifier: &Option<GroupModifier>,
    ) -> Signature {
        match modifier {
            None => self.signature(),
            Some(m) => match m.op {
                GroupModifierOp::On => self.signature_with_labels(m.labels()),
                GroupModifierOp::Ignoring => self.signature_without_labels(m.labels()),
            },
        }
    }
}

fn count_label_value(
    hm: &mut HashMap<String, HashMap<String, usize>>,
    label: &String,
    value: &String,
) {
    let label_counts = hm
        .entry(label.to_string())
        .or_insert_with(|| HashMap::new());
    *label_counts.entry(value.to_string()).or_insert(0) += 1;
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.metric_group)?;
        let len = self.tags.len();
        let mut i = 0;
        for Tag { key: k, value: v } in self.tags.iter() {
            write!(f, "{}={}", k, enquote('"', &v))?;
            if i < len - 1 {
                write!(f, ",")?;
            }
            i += 1;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

impl PartialOrd for MetricName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.metric_group != other.metric_group {
            return Some(self.metric_group.cmp(&other.metric_group));
        }
        // Metric names for a and b match. Compare tags.
        // Tags must be already sorted by the caller, so just compare them.
        for (a, b) in self.tags.iter().zip(&other.tags) {
            if let Some(ord) = a.partial_cmp(b) {
                if ord != Ordering::Equal {
                    return Some(ord);
                }
            }
        }

        return Some(self.tags.len().cmp(&other.tags.len()));
    }
}

pub struct WithLabelsIterator<'a> {
    tag_iter: std::slice::Iter<'a, Tag>,
    names_iter: std::slice::Iter<'a, String>,
}

impl<'a> WithLabelsIterator<'a> {
    pub fn new(metric_name: &'a MetricName, names: &'a [String]) -> Self {
        // todo: sort names here ?
        Self {
            names_iter: names.iter(),
            tag_iter: metric_name.tags.iter(),
        }
    }
}

impl<'a> Iterator for WithLabelsIterator<'a> {
    type Item = &'a Tag;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(tag) = self.tag_iter.next() {
            while let Some(name) = self.names_iter.next() {
                match name.cmp(&tag.key) {
                    Ordering::Less => {
                        continue;
                    }
                    Ordering::Equal => {
                        return Some(tag);
                    }
                    Ordering::Greater => {
                        break;
                    }
                }
            }
        }

        None
    }
}

pub struct WithoutLabelsIterator<'a> {
    tag_iter: std::slice::Iter<'a, Tag>,
    names: &'a [String],
    last_name: Option<&'a String>,
}

impl<'a> WithoutLabelsIterator<'a> {
    pub fn new(metric_name: &'a MetricName, names: &'a [String]) -> Self {
        Self {
            names,
            tag_iter: metric_name.tags.iter(),
            last_name: names.last(),
        }
    }
}

impl<'a> Iterator for WithoutLabelsIterator<'a> {
    type Item = &'a Tag;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(tag) = self.tag_iter.next() {
            if let Some(name) = self.last_name {
                if &tag.key > name {
                    return None;
                }
            }
            if !self.names.contains(&tag.key) {
                return Some(tag);
            }
        }
        None
    }
}
