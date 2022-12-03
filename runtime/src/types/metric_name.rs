use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Deref;

use enquote::enquote;
use xxhash_rust::xxh3::Xxh3;

use lib::{marshal_string_fast, unmarshal_string_fast, unmarshal_var_int};
use metricsql::common::{AggregateModifier, AggregateModifierOp, GroupModifier, GroupModifierOp};

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{marshal_bytes_fast, unmarshal_bytes_fast};
use serde::{Deserialize, Serialize};

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const METRIC_NAME_LABEL: &str = "__name__";

const ESCAPE_CHAR: u8 = 0_u8;
const TAG_SEPARATOR_CHAR: u8 = 1_u8;
const KV_SEPARATOR_CHAR: u8 = 2_u8;
const LABEL_SEP: u8 = 0xfe;
const SEP: u8 = 0xff;

/// Tag represents a (key, value) tag for metric.
#[derive(Debug, Default, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
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

    pub fn unmarshal<'a>(&mut self, src: &'a [u8]) -> RuntimeResult<&'a [u8]> {
        let mut src = src;
        match unmarshal_tag_value(src) {
            Ok((v, rest)) => {
                self.key = v.to_string();
                src = rest;
            }
            Err(_) => {
                return Err(RuntimeError::SerializationError(
                    "error reading tag key".to_string(),
                ));
            }
        }
        match unmarshal_tag_value(src) {
            Ok((v, rest)) => {
                self.key = v.to_string();
                src = rest;
            }
            Err(_) => {
                return Err(RuntimeError::SerializationError(
                    "error reading tag value".to_string(),
                ));
            }
        }
        Ok(src)
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

pub fn marshal_tag_value_no_trailing_tag_separator(dst: &mut Vec<u8>, src: &str) {
    marshal_tag_value(dst, src.as_bytes());
    // Remove trailing TAG_SEPARATOR_CHAR
    let _ = dst.remove(dst.len() - 1);
}

pub fn marshal_tag_value(dst: &mut Vec<u8>, src: &[u8]) {
    let n1 = src
        .iter()
        .position(|x| *x == ESCAPE_CHAR || *x == TAG_SEPARATOR_CHAR || *x == KV_SEPARATOR_CHAR);
    if n1.is_none() {
        // Fast path.
        dst.extend_from_slice(src);
        dst.push(TAG_SEPARATOR_CHAR);
    }
    // Slow path.
    for ch in src.iter() {
        match *ch {
            ESCAPE_CHAR => {
                dst.push(ESCAPE_CHAR);
                dst.push(0_u8);
            }
            TAG_SEPARATOR_CHAR => {
                dst.push(ESCAPE_CHAR);
                dst.push(1_u8);
            }
            KV_SEPARATOR_CHAR => {
                dst.push(ESCAPE_CHAR);
                dst.push(2_u8);
            }
            _ => dst.push(*ch),
        }
    }

    dst.push(TAG_SEPARATOR_CHAR)
}

#[inline]
fn u8_index_of(haystack: &[u8], needle: u8) -> Option<usize> {
    haystack.iter().position(|x| *x == needle)
}

// Todo(perf) use different serialization. This is just to make things work
pub fn unmarshal_tag_value<'a>(src: &'a [u8]) -> RuntimeResult<(String, &'a [u8])> {
    let n = u8_index_of(src, TAG_SEPARATOR_CHAR);
    if n.is_none() {
        return Err(RuntimeError::General(
            "cannot find the end of tag value".to_string(),
        ));
    }

    // todo: use a stack based buffer
    let mut dst: Vec<u8> = Vec::with_capacity(120); // ???

    let n = n.unwrap();
    let mut src = src;
    let mut b = &src[0..n];
    src = &src[n + 1..];

    loop {
        let n = u8_index_of(b, ESCAPE_CHAR);
        if n.is_none() {
            dst.extend_from_slice(b);
            return match String::from_utf8(dst) {
                Ok(v) => Ok((v, src)),
                Err(_) => Err(RuntimeError::General("invalid utf8 string".to_string())),
            };
        }

        let n = n.unwrap();
        dst.extend_from_slice(&b[0..n]);
        b = &b[n + 1..];
        if b.len() == 0 {
            return Err(RuntimeError::General("missing escaped char".to_string()));
        }
        let ch = b[0];
        match ch {
            0 => dst.push(ESCAPE_CHAR),
            1 => dst.push(TAG_SEPARATOR_CHAR),
            2 => dst.push(KV_SEPARATOR_CHAR),
            _ => {
                return Err(RuntimeError::General(format!(
                    "unsupported escaped char: {}",
                    ch
                )));
            }
        }
        b = &b[1..];
    }
}

/// MetricName represents a metric name.
#[derive(Debug, PartialEq, Eq, Clone, Default, Hash, Serialize, Deserialize)]
pub struct MetricName {
    pub metric_group: String,
    // todo: Consider https://crates.io/crates/btree-slab or heapless btree to minimize allocations
    pub tags: Vec<Tag>,
    pub(crate) hash: Option<u64>,
    sorted: bool,
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            metric_group: name.to_string(),
            tags: vec![],
            hash: None,
            sorted: false,
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
        self.sorted = false;
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
        let tag = Tag {
            key: key.to_string(),
            value: value.into(),
        };
        self.hash = None;
        self.sorted = false;
        self.tags.push(tag);
    }

    /// adds new tag to mn with the given key and value.
    pub fn set_tag<S: Into<String>>(&mut self, key: &str, value: S) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.into();
            self.hash = None;
            return;
        } else {
            for dst_tag in self.tags.iter_mut() {
                if dst_tag.key == key {
                    dst_tag.value = value.into();
                    self.hash = None;
                    return;
                }
            }
            let tag = Tag::new(key, value.into());
            self.tags.push(tag);
            self.sorted = false;
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
                self.sorted = false;
            }
        }
        self.hash = None;
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
            .iter()
            .find(|tag| tag.key == key)
            .and_then(|v| Some(&v.value))
    }

    pub fn without_keys_iter(&self, names: &[String]) -> impl Iterator<Item = &Tag> + '_ {
        // todo: optimize for small number of names
        let mut map: HashSet<String> = HashSet::with_capacity(names.len());
        for name in names {
            map.insert(name.to_string());
        }
        self.tags.iter().filter(move |tag| !map.contains(&tag.key))
    }

    pub fn with_keys_iter(&self, names: &[String]) -> impl Iterator<Item = &Tag> + '_ {
        // todo: optimize for small number of names
        let mut map: HashSet<String> = HashSet::with_capacity(names.len());
        for name in names {
            map.insert(name.to_string());
        }
        self.tags.iter().filter(move |tag| !map.contains(&tag.key))
    }

    /// remove_tags_on removes all the tags not included to onTags.
    pub fn remove_tags_on(&mut self, on_tags: &Vec<String>) {
        let set: HashSet<_> = HashSet::from_iter(on_tags);
        // written this way to avoid an allocation
        // (to compare against METRIC_NAME_LABEL.to_string())
        if !set.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_metric_group()
        }
        self.tags.retain(|tag| set.contains(&tag.key));
        self.sorted = false;
        self.hash = None;
    }

    /// remove_tags_ignoring removes all the tags included in ignoringTags.
    pub fn remove_tags_ignoring(&mut self, ignoring_tags: &Vec<String>) {
        let set: HashSet<_> = HashSet::from_iter(ignoring_tags);
        if set.iter().any(|x| x.as_str() == METRIC_NAME_LABEL) {
            self.reset_metric_group();
        }
        self.tags.retain(|tag| !set.contains(&tag.key));
        self.sorted = false;
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
        self.sorted = false;
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
        for Tag { key: k, value: v } in self.tags.iter() {
            marshal_bytes_fast(dst, k.as_bytes());
            marshal_bytes_fast(dst, v.as_bytes());
        }
    }

    /// marshal appends marshaled mn to dst.
    ///
    /// `self.sort_tags` must be called before calling this function
    /// in order to sort and de-duplicate tags.
    pub fn marshal(&self, dst: &mut Vec<u8>) {
        // Calculate the required size and pre-allocate space in dst
        let mut required_size = self.metric_group.len() + 1;
        for Tag { key: k, value: v } in self.tags.iter() {
            required_size += k.len() + v.len() + 2
        }

        dst.reserve(required_size);

        marshal_tag_value(dst, &self.metric_group.as_bytes());
        for Tag { key: k, value: v } in self.tags.iter() {
            required_size += k.len() + v.len() + 2
        }

        marshal_string_fast(dst, &self.metric_group);
        self.marshal_tags_fast(dst);
    }

    /// unmarshal unmarshals mn from src.
    /// Todo(perf) this is not necessarily as performant as can be, even for
    /// simple serialization
    pub fn unmarshal(&mut self, src: &[u8]) -> RuntimeResult<()> {
        // Unmarshal MetricGroup.
        let mut src = src;
        match unmarshal_tag_value(src) {
            Err(_) => {
                return Err(RuntimeError::SerializationError(format!(
                    "cannot unmarshal metric group"
                )))
            }
            Ok((str, tail)) => {
                src = tail;
                self.metric_group = str;
            }
        }
        while src.len() > 0 {
            let mut tag = Tag::default();
            src = tag.unmarshal(src)?;
            self.tags.push(tag);
        }
        Ok(())
    }

    /// marshal_raw marshals mn to dst and returns the result.
    ///
    /// The results may be unmarshalled with MetricName.UnmarshalRaw.
    ///
    /// This function is for testing purposes. MarshalMetricNameRaw must be used
    /// in prod instead.
    pub(crate) fn marshal_raw(&mut self, dst: &mut Vec<u8>) {
        marshal_bytes_fast(dst, &self.metric_group.as_bytes());
        self.sort_tags();
        for tag in self.tags.iter() {
            marshal_bytes_fast(dst, tag.key.as_bytes());
            marshal_bytes_fast(dst, &tag.value.as_bytes());
        }
    }

    /// UnmarshalRaw unmarshals mn encoded with MarshalMetricNameRaw.
    pub(crate) fn unmarshal_raw(&mut self, src: &[u8]) -> RuntimeResult<()> {
        self.reset();
        let mut src = src;
        while src.len() > 0 {
            let (tail, key) = unmarshal_bytes_fast(src)?;
            src = tail;

            let (tail, value) = unmarshal_bytes_fast(src)?;
            src = tail;

            let val = String::from_utf8_lossy(value);
            if key.len() == 0 {
                self.metric_group = val.to_string();
            } else {
                let key = String::from_utf8_lossy(key);
                self.add_tag(&key, val);
            }
        }
        Ok(())
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
                return Err(RuntimeError::SerializationError(format!(
                    "cannot unmarshal MetricGroup: {:?}",
                    err
                )));
            }
            Ok((tail, metric_group)) => {
                src = tail;
                self.metric_group = String::from_utf8_lossy(metric_group).to_string();
            }
        }

        if src.len() < 2 {
            return Err(RuntimeError::SerializationError(format!(
                "not enough bytes for unmarshalling len(tags); need at least 2 bytes; got {} bytes",
                src.len()
            )));
        }

        let tags_len: u16;

        match unmarshal_var_int::<u16>(src) {
            Ok((len, tail)) => {
                src = tail;
                tags_len = len;
            }
            Err(err) => {
                return Err(RuntimeError::SerializationError(format!(
                    "error reading tags length: {}",
                    err
                )));
            }
        }

        let mut key: String;
        let mut val: String;

        for i in 0..tags_len {
            match unmarshal_string_fast(&mut src) {
                Err(_) => {
                    return Err(RuntimeError::SerializationError(format!(
                        "cannot unmarshal key for tag[{}]",
                        i
                    )));
                }
                Ok((t, v)) => {
                    src = v;
                    key = t;
                }
            }

            match unmarshal_string_fast(&mut src) {
                Err(_) => {
                    return Err(RuntimeError::SerializationError(format!(
                        "cannot unmarshal value for tag[{}]",
                        i
                    )));
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
                self.remove_tags(labels);
                // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                self.reset_metric_group();
            }
        }
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

    pub fn fast_hash(&self) -> u64 {
        let mut hasher: Xxh3 = Xxh3::new();
        hasher.update(self.metric_group.as_bytes());
        for Tag { key: k, value: v } in self.tags.iter() {
            hasher.update(k.as_bytes());
            hasher.update(v.as_bytes());
        }
        hasher.digest()
    }

    pub fn get_hash(&mut self) -> u64 {
        self.hash.unwrap_or_else(|| {
            self.sort_tags();
            let hash = self.fast_hash();
            self.hash = Some(hash);
            hash
        })
    }

    pub fn sort_tags(&mut self) {
        if self.sorted {
            return;
        }
        self.tags.sort_by(|a, b| {
            if a.key == b.key {
                return a.value.cmp(&b.value);
            }
            return a.key.cmp(&b.key);
        });
        self.sorted = true;
    }

    pub fn to_string(&self) -> String {
        // todo(perf): preallocate result buf
        let mut result = String::new();
        result.push_str(&self.metric_group);
        result.push_str("{");
        let len = self.tags.len();
        let mut i = 0;
        for Tag { key: k, value: v } in self.tags.iter() {
            result.push_str(format!("{}={}", k, enquote('"', &v)).as_str());
            if i < len - 1 {
                result.push_str(",");
            }
            i += 1;
        }
        result.push_str("}");
        result
    }

    pub fn hash_with_labels(&self, buf: &mut Vec<u8>, hasher: &mut Xxh3, names: &[String]) -> u64 {
        hasher.reset();
        self.bytes_with_labels(buf, names);
        hasher.update(buf);
        hasher.digest()
    }

    pub fn hash_without_labels(&self, buf: &mut Vec<u8>, hasher: &mut Xxh3, names: &[String]) -> u64 {
        hasher.reset();
        self.bytes_without_labels(buf, names);
        hasher.update(buf);
        hasher.digest()
    }

    // bytes_without_labels is just as bytes(), but only for labels not matching names.
    // 'names' have to be sorted in ascending order.
    pub fn bytes_without_labels(&self, buf: &mut Vec<u8>, names: &[String]) {
        buf.clear();
        buf.push(LABEL_SEP);

        let mut names_iter = names.iter();
        let mut ls_iter = self.tags.iter();

        while let Some(label) = ls_iter.next() {
            while let Some(name) = names_iter.next() {
                if *name < label.key {
                    names_iter.next();
                } else if *name == label.key {
                    continue;
                } else {
                    break;
                }
            }

            add_tag_to_buf(buf, label);
        }
    }

    // bytes_with_labels is just like Bytes(), but only for labels matching names.
    // 'names' have to be sorted in ascending order.
    pub fn bytes_with_labels(&self, buf: &mut Vec<u8>, names: &[String]) {
        buf.clear();
        buf.push(LABEL_SEP);
        let mut ts_iter = self.tags.iter();
        let mut names_iter = names.iter();

        let mut tag = ts_iter.next();
        let mut name = names_iter.next();
        loop {
            match (tag, name) {
                (Some(_tag), Some(_name)) => {
                    if *_name < _tag.key {
                        name = names_iter.next();
                    } else if _tag.key < *_name {
                        tag = ts_iter.next();
                    } else {
                        add_tag_to_buf(buf, _tag);
                        name = names_iter.next();
                        tag = ts_iter.next();
                    }
                }
                _ => {
                    break
                }
            }
        }
    }
}

fn add_tag_to_buf(b: &mut Vec<u8>, tag: &Tag) {
    if b.len() > 1 {
        b.push(SEP)
    }
    b.extend_from_slice(&tag.key.as_bytes());
    b.push(SEP);
    b.extend_from_slice(&tag.value.as_bytes());
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
            if a.key != b.key {
                return Some(a.key.cmp(&b.key));
            }
            if a.value != b.value {
                return Some(a.value.cmp(&b.value));
            }
        }

        return Some(self.tags.len().cmp(&other.tags.len()));
    }
}
