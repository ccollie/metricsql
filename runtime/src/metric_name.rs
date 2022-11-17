use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::btree_map::Iter;
use std::fmt;
use std::fmt::Display;
use std::hash::{Hash};
use std::ops::{Deref};

use enquote::enquote;
use xxhash_rust::xxh3::Xxh3;

use lib::{marshal_string_fast, unmarshal_string_fast, unmarshal_var_int};
use metricsql::ast::{AggregateModifier, AggregateModifierOp, GroupModifier, GroupModifierOp};

use crate::{marshal_bytes_fast, unmarshal_bytes_fast};
use crate::runtime_error::{RuntimeError, RuntimeResult};

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const METRIC_NAME_LABEL: &str = "__name__";

const LABEL_SEP: u8 = 0xfe;
const SEPS: &[u8; 1] = b"\xff";

/// Tag represents a (key, value) tag for metric.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Tag {
    pub(crate) key: String,
    pub(crate) value: String,
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
#[derive(Debug, PartialEq, Eq, Clone, Default, Hash)]
pub struct MetricName {
    pub metric_group: String,
    // todo: Consider https://crates.io/crates/btree-slab or heapless btree to minimize allocations
    items: BTreeMap<String, String>,
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            metric_group: name.to_string(),
            items: BTreeMap::new(),
        }
    }

    /// from_strings creates new labels from pairs of strings.
    pub fn from_strings(ss: &[&str]) -> RuntimeResult<Self> {
        if ss.len() %2 != 0 {
            return Err(RuntimeError::from("invalid number of strings"));
        }

        let mut res = MetricName::default();
        for i in (0 .. ss.len()).step_by(2) {
            res.set_tag(ss[i], ss[i+1]);
        }

        Ok(res)
    }

    pub fn reset_metric_group(&mut self) {
        self.metric_group = "".to_string();
    }

    pub fn tag_count(&self) -> usize {
        self.items.len()
    }

    pub fn copy_from(&mut self, other: &MetricName) {
        self.metric_group = other.metric_group.clone();
        // todo: can we make this more efficient ?
        self.items.clear();
        self.items.clone_from(&other.items);
    }

    /// Reset resets the mn.
    pub fn reset(&mut self) {
        self.metric_group = "".to_string();
        self.items.clear();
    }

    pub fn set_metric_group(&mut self, value: &str) {
        self.metric_group = value.to_string();
    }

    /// adds new tag to mn with the given key and value.
    pub fn set_tag(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.to_string();
            return;
        }
        self.items.insert(key.to_string(), value.to_string());
    }

    /// removes a tag with the given tagKey
    pub fn remove_tag(&mut self, key: &str) {
        if key == METRIC_NAME_LABEL {
            self.reset_metric_group();
            return;
        }
        self.items.remove(key);
    }

    /// replaces a tag value
    pub fn replace_tag(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.to_string();
            return;
        }
        self.items.insert(key.to_string(), value.to_string());
    }

    // todo: rewrite to pass references
    pub fn get_tags(&self) -> Vec<Tag> {
        self.items.iter().map(|(k, v)| {
            Tag {
                key: k.to_string(),
                value: v.to_string(),
            }
        }).collect()
    }

    pub fn for_each_tag<F>(&self, mut f: F) -> ()
    where F: FnMut(&String, &String) -> () {
        if self.metric_group.len() > 0 {
            f(&METRIC_NAME_LABEL.to_string(), &self.metric_group);
        }
        self.items.iter().for_each(|(k,v)| f(k, v))
    }

    pub fn has_tag(&self, key: &str) -> bool {
        self.items.contains_key(key)
    }

    /// returns tag value for the given tagKey.
    pub fn get_tag_value(&self, key: &str) -> Option<&String> {
        if key == METRIC_NAME_LABEL {
            return Some(&self.metric_group);
        }
        self.items.get(key)
    }

    pub fn get_tag_value_mut(&mut self, key: &str) -> Option<&mut String> {
        if key == METRIC_NAME_LABEL {
            return Some(&mut self.metric_group);
        }
        self.items.get_mut(key)
    }

    /// removes all the tags not included in on_tags.
    pub fn remove_tags_except(&mut self, tags: &[String]) {
        let set: HashSet<&String> = HashSet::from_iter(tags);
        if set.contains(&METRIC_NAME_LABEL.to_string()) {
            self.reset_metric_group()
        }
        self.items.retain(|k, _| set.contains(k));
    }

    pub fn remove_tags_on(&mut self, on_tags: &[String]) {
        self.remove_tags_except(on_tags)
    }

    pub fn update_tags_by_group_modifier(&mut self, modifier: &GroupModifier) {
        match modifier.op {
            GroupModifierOp::On => {
                self.remove_tags_except(&modifier.labels);
            },
            GroupModifierOp::Ignoring => {
                self.remove_tags(&modifier.labels);
            },
        }
    }
    
    /// removes all the tags included in ignoring_tags.
    pub fn remove_tags(&mut self, ignoring_tags: &[String]) {
        for tag in ignoring_tags {
            if tag == METRIC_NAME_LABEL {
                self.reset_metric_group();
            } else {
                self.items.remove(tag);
            }
        }
    }

    /// sets tags from src with keys matching add_tags.
    pub(crate) fn set_tags(&mut self, add_tags: &[String], src: &mut MetricName) {
        for tag_name in add_tags {
            if tag_name == METRIC_NAME_LABEL {
                self.metric_group = tag_name.clone();
                continue;
            }

            match src.get_tag_value(tag_name) {
                Some(tag_value) => {
                    let _ = self.items.insert(tag_name.to_string(), tag_value.to_string());
                },
                None => {
                    self.remove_tag(tag_name);
                }
            }
        }
    }

    pub fn append_tags_to_string(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice("{{".as_bytes());

        let len = &self.items.len();
        let mut i = 0;
        for (k, v) in &self.items {
            dst.extend_from_slice(format!("{}={}", k, enquote('"', &v)).as_bytes());
            if i + 1 < *len {
                dst.extend_from_slice(", ".as_bytes())
            }
            i += 1;
        }
        // ??????????????
        dst.extend_from_slice('}'.to_string().as_bytes());
    }

    pub(crate) fn get_bytes(&self, dst: &mut Vec<u8>) {

    }

    pub(crate) fn marshal_tags_fast(&self, dst: &mut Vec<u8>) {
        for (k, v) in &self.items {
            marshal_bytes_fast(dst, k.as_bytes());
            marshal_bytes_fast(dst, v.as_bytes());
        }
    }

    pub fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_string_fast(dst, &self.metric_group);
        self.marshal_tags_fast(dst);
    }


    /// unmarshal mn from src, so mn members hold references to src.
    ///
    /// It is unsafe modifying src while mn is in use.
    pub fn unmarshal_fast(src: &[u8]) -> RuntimeResult<(MetricName, &[u8])> {
        let mut mn: MetricName = MetricName::default();
        let tail = mn.unmarshal_fast_internal(src)?;
        return Ok((mn, tail));
    }

    /// unmarshal mn from src
    pub(crate) fn unmarshal_fast_internal<'a>(&mut self, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
        let mut src = src;

        match unmarshal_bytes_fast(src) {
            Err(err) => {
                return Err(RuntimeError::SerializationError(format!("cannot unmarshal MetricGroup: {:?}", err)));
            }
            Ok((tail, metric_group)) => {
                src = tail;
                self.metric_group = String::from_utf8_lossy(metric_group).to_string();
            }
        }

        if src.len() < 2 {
            return Err(RuntimeError::SerializationError(
                format!("not enough bytes for unmarshalling len(tags); need at least 2 bytes; got {} bytes", src.len())
            ));
        }

        let tags_len: u16;

        match unmarshal_var_int::<u16>(src) {
            Ok((len, tail)) => {
                src = tail;
                tags_len = len;
            },
            Err(err) => {
                return Err(RuntimeError::SerializationError(format!("error reading tags length: {}", err)));
            }
        }

        let mut key: String;
        let mut val: String;

        for i in 0..tags_len {
            match unmarshal_string_fast(&mut src) {
                Err(_) => {
                    return Err(RuntimeError::SerializationError(format!("cannot unmarshal key for tag[{}]", i)));
                }
                Ok((t, v)) => {
                    src = v;
                    key = t;
                }
            }

            match unmarshal_string_fast(&mut src) {
                Err(_) => {
                    return Err(
                        RuntimeError::SerializationError(format!("cannot unmarshal value for tag[{}]", i))
                    );
                }
                Ok((t, v)) => {
                    src = v;
                    val = t;
                }
            }

            self.set_tag(&key, &val);
        }

        Ok(src)
    }

    pub fn iter(&self) -> Iter<'_, String, String> {
        self.items.iter()
    }

    pub(crate) fn serialized_size(&self) -> usize {
        let mut n = 2 + self.metric_group.len();
        n += 2; // Length of tags.
        for (k,v) in self.items.iter() {
            n += 2 + k.len();
            n += 2 + v.len();
        }
        n
    }

    pub fn remove_group_tags(&mut self, modifier: &Option<AggregateModifier>) {
        let mut group_op = AggregateModifierOp::By;
        let mut labels = &vec![]; // zero alloc

        if let Some(m) = modifier.deref() {
            group_op = m.op.clone();
            labels = &m.args
        };

        match group_op {
            AggregateModifierOp::By => {
                self.remove_tags_except(labels);
            }
            AggregateModifierOp::Without => {
                self.remove_tags(labels);
                // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                self.reset_metric_group();
            }
        }
    }

    pub(crate) fn count_label_values(&self, hm: &mut HashMap<String, HashMap<String, usize>>) {
       // duplication, I know
        let label_counts = hm.entry(METRIC_NAME_LABEL.to_string())
            .or_insert_with(|| HashMap::new());
        *label_counts.entry(self.metric_group.to_string()).or_insert(0) += 1;
        for entry in self.items.iter() {
            count_label_value(hm, entry.0, entry.1);
       }
    }

    pub fn fast_hash(&self) -> u64 {
        let mut hasher: Xxh3 = Xxh3::new();
        hasher.update(self.metric_group.as_bytes());
        for (k, v) in self.items.iter() {
            hasher.update(k.as_bytes());
            hasher.update(v.as_bytes());
        }
        hasher.digest()
    }

    pub fn to_string(&self) -> String {
        // todo(perf): preallocate result buf
        let mut result = String::new();
        result.push_str(&self.metric_group);
        result.push_str("{{");
        let len = self.items.len();
        let mut i = 0;
        for (k, v) in self.items.iter() {
            result.push_str(format!("{}={}", k, enquote('"', &v)).as_str());
            if i < len - 1 {
                result.push_str(",");
            }
            i += 1;
        }
        result.push_str("}}");
        result
    }
}

fn count_label_value(hm: &mut HashMap<String, HashMap<String, usize>>, label: &String, value: &String) {
    let label_counts = hm.entry(label.to_string())
        .or_insert_with(|| HashMap::new());
    *label_counts.entry(value.to_string()).or_insert(0) += 1;
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.metric_group)?;
        let len = self.items.len();
        let mut i = 0;
        for (k, v) in self.items.iter() {
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
        let ats = &self.items;
        let bts = &other.items;
        let mut b_iter = other.items.iter();
        for (a_key, a_value) in self.items.iter() {
            let bt = &b_iter.next();
            if bt.is_none() {
                // a contains more tags than b and all the previous tags were identical,
                // so a is considered bigger than b.
                return Some(Ordering::Greater);
            }
            let (b_key, b_value) = bt.unwrap();
            if a_key != b_key {
                return Some(a_key.cmp(b_key));
            }
            if a_value != b_value {
                return Some(a_value.cmp(b_value));
            }
        }
        return Some(ats.len().cmp(&bts.len()));
    }
}
