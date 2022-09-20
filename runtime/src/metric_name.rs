use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::collections::btree_map::Iter;
use std::fmt;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

use enquote::enquote;
use lockfree_object_pool::{LinearObjectPool, LinearReusable};
use once_cell::sync::Lazy;

use lib::{unmarshal_string_fast, unmarshal_var_int};
use metricsql::ast::{AggregateModifier, AggregateModifierOp};

use crate::{marshal_bytes_fast, marshal_string_fast, unmarshal_bytes_fast};
use crate::runtime_error::{RuntimeError, RuntimeResult};

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const NAME_LABEL: &str = "__name__";

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
    // todo: Consider https://crates.io/crates/btree-slab to minimize allocations
    _items: BTreeMap<String, String>,
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            metric_group: name.to_string(),
            _items: BTreeMap::new(),
        }
    }

    pub fn reset_metric_group(&mut self) {
        self.metric_group = "".to_string();
    }

    pub fn get_tag_count(self) -> usize {
        self._items.len()
    }

    pub fn copy_from(&mut self, other: &MetricName) {
        self.metric_group = other.metric_group.clone();
        // todo: can we make this more efficient ?
        self._items.clear();
        self._items.clone_from(&other._items);
    }

    /// Reset resets the mn.
    pub fn reset(&mut self) {
        self.metric_group = "".to_string();
        self._items.clear();
    }

    /// adds new tag to mn with the given key and value.
    pub fn add_tag(&mut self, key: &str, value: &str) {
        if key == NAME_LABEL {
            self.metric_group = value.to_string();
            return;
        }
        self._items.insert(key.to_string(), value.to_string());
    }

    /// removes a tag with the given tagKey
    pub fn remove_tag(&mut self, key: &str) {
        if key == NAME_LABEL {
            self.reset_metric_group();
            return;
        }
        self._items.remove(key);
    }

    /// replaces a tag value
    pub fn replace_tag(&mut self, key: &str, value: &str) {
        if key == NAME_LABEL {
            self.metric_group = value.to_string();
            return;
        }
        self._items.insert(key.to_string(), value.to_string());
    }

    // todo: rewrite to pass references
    pub fn get_tags(&self) -> Vec<Tag> {
        self._items.iter().map(|(k, v)| {
            Tag {
                key: k.to_string(),
                value: v.to_string(),
            }
        }).collect()
    }

    pub fn has_tag(&self, key: &str) -> bool {
        self._items.contains_key(key)
    }

    /// returns tag value for the given tagKey.
    pub fn get_tag_value(&self, key: &str) -> Option<&String> {
        if key == NAME_LABEL {
            return Some(&self.metric_group);
        }
        self._items.get(key)
    }

    pub fn get_tag_value_mut(&mut self, key: &str) -> Option<&mut String> {
        if key == NAME_LABEL {
            return Some(&mut self.metric_group);
        }
        self._items.get_mut(key)
    }

    /// removes all the tags not included to on_tags.
    pub fn remove_tags_on(&mut self, on_tags: &[String]) {
        let set: HashSet<&String> = HashSet::from_iter(on_tags);
        if set.contains(&NAME_LABEL.to_string()) {
            self.reset_metric_group()
        }
        self._items.retain(|k, _| set.contains(k));
    }

    /// removes all the tags included in ignoring_tags.
    pub fn remove_tags_ignoring(&mut self, ignoring_tags: &[String]) {
        for tag in ignoring_tags {
            if tag == NAME_LABEL {
                self.metric_group = "".to_string();
            } else {
                self._items.remove(tag);
            }
        }
    }

    /// sets tags from src with keys matching add_tags.
    pub(crate) fn set_tags(&mut self, add_tags: &[String], mut src: &MetricName) {
        for tag_name in add_tags {
            if tag_name == NAME_LABEL {
                self.metric_group = tag_name.clone();
                continue;
            }

            match src.get_tag_value(tag_name) {
                Some(tag_value) => {
                    if let Some(mut it) = self._items.get_mut(tag_name) {
                        it = &mut tag_value.to_string()
                    } else {
                        self._items.insert(tag_name.to_string(), tag_value.to_string());
                    }
                },
                None => {
                    self.remove_tag(tag_name);
                }
            }
        }
    }

    pub fn append_tags_to_string(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice("{{".as_bytes());

        let len = self._items.len();
        let mut i = 0;
        for (k, v) in self._items {
            dst.extend_from_slice(format!("{}={}", k, enquote('"', &v)).as_bytes());
            if i + 1 < len {
                dst.extend_from_slice(", ".as_bytes())
            }
            i += 1;
        }
        // ??????????????
        dst.extend_from_slice('}'.to_string().as_bytes());
    }

    pub(crate) fn marshal_tags_fast(&self, dst: &mut Vec<u8>) {
        for (k, v) in self._items {
            marshal_bytes_fast(dst, k.as_bytes());
            marshal_bytes_fast(dst, v.as_bytes());
        }
    }

    pub fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_string_fast(dst, &self.metric_group);
        self.marshal_tags_fast(dst);
    }


    // internal only
    pub(crate) fn marshal_to_string(&self, buf: &mut Vec<u8>) -> Cow<'_, str> {
        self.marshal(buf);
        String::from_utf8_lossy(&buf)
    }

    /// unmarshal mn from src, so mn members hold references to src.
    ///
    /// It is unsafe modifying src while mn is in use.
    pub fn unmarshal_fast(src: &[u8]) -> RuntimeResult<(MetricName, &[u8])> {
        let mut mn: MetricName = MetricName::default();
        let tail = mn.unmarshal_fast_internal(src)?;
        return Ok((mn, tail));
    }

    /// unmarshal mn from src, so mn members hold references to src.
    ///
    /// It is unsafe modifying src while mn is in use.
    pub(crate) fn unmarshal_fast_internal(&mut self, src: &[u8]) -> RuntimeResult<&[u8]> {
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

        let mut tags_len: u16 = 0;

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

            self.add_tag(&key, &val);
        }

        Ok(src)
    }

    pub fn iter(&self) -> Iter<'_, String, String> {
        self._items.iter()
    }

    pub(crate) fn serialized_size(self) -> usize {
        let mut n = 2 + self.metric_group.len();
        n += 2; // Length of tags.
        for (k,v) in self._items.iter() {
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
                self.remove_tags_on(labels);
            }
            AggregateModifierOp::Without => {
                self.remove_tags_ignoring(labels);
                // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                self.reset_metric_group();
            }
        }
    }
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.metric_group)?;
        let len = self._items.len();
        let mut i = 0;
        for (k, v) in self._items {
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
        let ats = &self._items;
        let bts = &other._items;
        let mut b_iter = other._items.iter();
        for (a_key, a_value) in ats.iter() {
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

static METRICNAME_POOL: Lazy<LinearObjectPool<MetricName>> = Lazy::new(||{
    LinearObjectPool::<MetricName>::new(
        || MetricName::default(),
        |v| { v.reset(); }
    )
});

pub(crate) fn get_pooled_metric_name() -> &'static MetricName {
    LinearReusable::deref_mut(&mut METRICNAME_POOL.pull())
}