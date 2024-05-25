use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use ahash::{AHashMap, AHashSet};
use enquote::enquote;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use metricsql_parser::label::LabelFilterOp;
use metricsql_parser::prelude::{AggregateModifier, VectorMatchModifier};

use crate::common::encoding::{read_string, read_usize, write_string, write_usize};
use crate::parse_metric_selector;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::signature::Signature;

/// The maximum length of label name.
///
/// Longer names are truncated.
pub const MAX_LABEL_NAME_LEN: usize = 256;

pub const METRIC_NAME_LABEL: &str = "__name__";

const SEP: u8 = 0xff;

// for tag manipulation (removing, adding, etc.), name vectors longer than this will be converted to a hashmap
// for comparison, otherwise we do a linear probe
const SET_SEARCH_MIN_THRESHOLD: usize = 16;

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
            value,
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
}

impl MetricName {
    pub fn new(name: &str) -> Self {
        MetricName {
            metric_group: name.to_string(),
            tags: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.metric_group.is_empty() && self.tags.is_empty()
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

    pub fn parse(s: &str) -> RuntimeResult<Self> {
        let filters = parse_metric_selector(s)?;
        let mut mn = MetricName::default();
        // make sure we only have '=' filters
        for f in filters.into_iter() {
            if f.op != LabelFilterOp::Equal {
                return Err(RuntimeError::from(format!(
                    "invalid operator {} in metric name",
                    f.op
                )));
            }
            let tag = Tag {
                key: f.label,
                value: f.value,
            };
            mn.tags.push(tag);
        }
        mn.sort_tags();
        Ok(mn)
    }

    pub fn reset_metric_group(&mut self) {
        self.metric_group.clear();
    }

    pub fn copy_from(&mut self, other: &MetricName) {
        self.metric_group = other.metric_group.clone();
        self.tags = other.tags.clone();
    }

    /// Reset resets the mn.
    pub fn reset(&mut self) {
        self.metric_group = "".to_string();
        self.tags.clear();
    }

    pub fn set_metric_group(&mut self, value: &str) {
        self.metric_group = value.to_string();
    }

    /// add_tag adds new tag to mn with the given key and value.
    pub fn add_tag(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.into();
            return;
        }
        self.upsert(key, value);
    }

    pub fn add_tags_from_hashmap(&mut self, tags: &HashMap<String, String>) {
        for (key, value) in tags {
            if key == METRIC_NAME_LABEL {
                self.metric_group = value.into();
                return;
            }
            self.upsert(key, value);
        }
    }

    fn upsert(&mut self, key: &str, value: &str) {
        match self.tags.binary_search_by_key(&key, |tag| &tag.key) {
            Ok(idx) => {
                let tag = &mut self.tags[idx];
                tag.value.clear();
                tag.value.push_str(value);
            }
            Err(idx) => {
                let tag = Tag {
                    key: key.to_string(),
                    value: value.to_string(),
                };
                self.tags.insert(idx, tag);
            }
        }
    }

    /// adds new tag to mn with the given key and value.
    pub fn set_tag(&mut self, key: &str, value: &str) {
        if key == METRIC_NAME_LABEL {
            self.metric_group = value.into();
        } else {
            self.upsert(key, value);
        }
    }

    /// removes a tag with the given tagKey
    pub fn remove_tag(&mut self, key: &str) {
        if key == METRIC_NAME_LABEL {
            self.reset_metric_group();
        } else {
            self.tags.retain(|x| x.key != key);
        }
    }

    pub fn has_tag(&self, key: &str) -> bool {
        self.tags.iter().any(|x| x.key == key)
    }

    fn get_tag_index(&self, key: &str) -> Option<usize> {
        if self.tags.is_empty() {
            return None;
        }
        if self.tags.len() < 8 {
            return self.tags.iter().position(|x| x.key == key);
        }
        self.tags.binary_search_by_key(&key, |tag| &tag.key).ok()
    }

    /// returns tag value for the given tagKey.
    pub fn tag_value(&self, key: &str) -> Option<&String> {
        if key == METRIC_NAME_LABEL {
            return Some(&self.metric_group);
        }
        if let Some(index) = self.get_tag_index(key) {
            return Some(&self.tags[index].value);
        }
        None
    }

    #[allow(unused)]
    pub(crate) fn get_value_mut(&mut self, name: &str) -> Option<&mut String> {
        if name == METRIC_NAME_LABEL {
            return Some(&mut self.metric_group);
        }
        if let Some(index) = self.get_tag_index(name) {
            return Some(&mut self.tags[index].value);
        }
        None
    }

    /// remove_tags_on removes all the tags not included in on_tags.
    /// don't stare too deeply. Just convince yourself that this is the correct behavior.
    /// https://github.com/VictoriaMetrics/VictoriaMetrics/blob/cde5029bcecac116b59e245330f6caf625e75eea/lib/storage/metric_name.go#L247
    pub fn remove_tags_on(&mut self, on_tags: &[String]) {
        if !on_tags.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_metric_group()
        }
        if on_tags.is_empty() {
            self.tags.clear();
            return;
        }
        if on_tags.len() > SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(on_tags);
            self.tags.retain(|tag| set.contains(&tag.key));
        } else {
            self.tags.retain(|tag| on_tags.contains(&tag.key));
        }
    }

    /// remove_tags_ignoring removes all the tags included in ignoring_tags.
    pub fn remove_tags_ignoring(&mut self, ignoring_tags: &[String]) {
        self.remove_tags(ignoring_tags);
    }

    /// removes all the tags included in labels.
    pub fn remove_tags(&mut self, labels: &[String]) {
        if labels.is_empty() {
            return;
        }
        if labels.iter().any(|x| x.as_str() == METRIC_NAME_LABEL) {
            self.reset_metric_group();
        }

        if labels.len() > SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(labels);
            self.tags.retain(|tag| !set.contains(&tag.key));
        } else {
            self.tags.retain(|tag| !labels.contains(&tag.key));
        }
    }

    pub fn retain_tags(&mut self, tags: &[String]) {
        if !tags.iter().any(|x| *x == METRIC_NAME_LABEL) {
            self.reset_metric_group()
        }
        if tags.is_empty() {
            self.tags.clear();
            return;
        }
        if tags.len() >= SET_SEARCH_MIN_THRESHOLD {
            let set: AHashSet<_> = AHashSet::from_iter(tags);
            self.tags.retain(|tag| set.contains(&tag.key));
        } else {
            self.tags.retain(|tag| tags.contains(&tag.key));
        }
    }

    /// sets tags from src with keys matching add_tags.
    pub(crate) fn set_tags(
        &mut self,
        prefix: &str,
        add_tags: &[String],
        skip_tags: &[String],
        src: &mut MetricName,
    ) {
        if add_tags.len() == 1 && add_tags[0] == "*" {
            // Special case for copying all the tags except of skipTags from src to mn.
            self.set_all_tags(prefix, skip_tags, src);
            return;
        }

        for tag_name in add_tags {
            if skip_tags.contains(tag_name) {
                continue;
            }

            // todo: use iterators instead
            match src.tag_value(tag_name) {
                Some(tag_value) => {
                    if !prefix.is_empty() {
                        let key = format!("{prefix}{}", tag_name);
                        self.set_tag(&key, tag_value);
                    } else {
                        self.set_tag(tag_name, tag_value);
                    }
                }
                None => {
                    self.remove_tag(tag_name);
                }
            }
        }
    }

    fn set_all_tags(&mut self, prefix: &str, skip_tags: &[String], src: &MetricName) {
        for tag in src.tags.iter() {
            if skip_tags.contains(&tag.key) {
                continue;
            }
            if !prefix.is_empty() {
                let key = format!("{prefix}{}", tag.key);
                self.set_tag(&key, &tag.value);
            } else {
                self.set_tag(&tag.key, &tag.value);
            }
        }
    }

    pub fn append_tags_to_string(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice("{{".as_bytes());

        let len = &self.tags.len();
        for (i, Tag { key: k, value: v }) in self.tags.iter().enumerate() {
            dst.extend_from_slice(format!("{}={}", k, enquote('"', v)).as_bytes());
            if i + 1 < *len {
                dst.extend_from_slice(", ".as_bytes())
            }
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

    /// unmarshals mn from src.
    /// Todo(perf) this is not necessarily as performant as can be, even for
    /// simple serialization
    pub fn unmarshal(src: &[u8]) -> RuntimeResult<(&[u8], MetricName)> {
        let mut mn = MetricName::default();
        let (mut src, group) = read_string(src, "metric group")?;

        mn.metric_group = group;
        src = mn.unmarshal_tags(src)?;
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
        if let Some(m) = modifier {
            match m {
                AggregateModifier::By(labels) => {
                    // we're grouping by `labels, so keep only those
                    self.retain_tags(labels);
                }
                AggregateModifier::Without(labels) => {
                    self.remove_tags(labels);
                    // Reset metric group as Prometheus does on `aggr(...) without (...)` call.
                    self.reset_metric_group();
                }
            }
        } else {
            // No grouping. Remove all tags.
            self.remove_tags_on(&[]);
        };
    }

    pub(crate) fn count_label_values(&self, hm: &mut AHashMap<String, AHashMap<String, usize>>) {
        // duplication, I know
        let label_counts = hm.entry(METRIC_NAME_LABEL.to_string()).or_default();
        *label_counts
            .entry(self.metric_group.to_string())
            .or_insert(0) += 1;
        for tag in self.tags.iter() {
            count_label_value(hm, &tag.key, &tag.value);
        }
    }

    pub(crate) fn signature(&self) -> Signature {
        Signature::new(self)
    }

    pub fn sort_tags(&mut self) {
        if self.tags.len() > 1 {
            self.tags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    /// generate a Signature using tags only (excluding the metric group)
    pub fn tags_signature(&self) -> Signature {
        Signature::from_tags(self)
    }

    /// Compute a signature hash using only the tags passed in `labels`. `metric_group` is ignored.
    pub fn tags_signature_with_labels(&self, labels: &[String]) -> Signature {
        let empty: &str = "";
        self.signature_with_labels_internal(empty, labels)
    }

    /// Compute a signature hash using only the tags passed in `labels`.
    pub fn signature_with_labels(&self, labels: &[String]) -> Signature {
        self.signature_with_labels_internal(&self.metric_group, labels)
    }

    fn signature_with_labels_internal(&self, name: &str, labels: &[String]) -> Signature {
        let iter = self.tags.iter().filter(|tag| labels.contains(&tag.key));
        Signature::with_name_and_labels(name, iter)
    }

    /// Compute a signature hash ignoring the passed in labels.
    pub fn signature_without_labels(&self, labels: &[String]) -> Signature {
        self.signature_with_labels_internal(&self.metric_group, labels)
    }

    /// Compute a signature hash from tags ignoring the passed in labels.
    /// The `metric_group` is ignored.
    pub fn tags_signature_without_labels(&self, labels: &[String]) -> Signature {
        self.signature_without_labels_internal("", labels)
    }

    fn signature_without_labels_internal(&self, name: &str, labels: &[String]) -> Signature {
        if labels.is_empty() {
            let iter = self.tags.iter();
            return Signature::with_name_and_labels(name, iter);
        }
        let includes_metric_group = labels.iter().any(|x| *x == METRIC_NAME_LABEL);
        let group_name = if includes_metric_group { name } else { "" };
        let iter = self.tags.iter().filter(|tag| !labels.contains(&tag.key));
        Signature::with_name_and_labels(group_name, iter)
    }

    /// Calculate signature for the metric name by the given match modifier.
    pub fn signature_by_match_modifier(&self, modifier: &Option<VectorMatchModifier>) -> Signature {
        match modifier {
            None => self.signature(),
            Some(m) => match m {
                VectorMatchModifier::On(labels) => self.signature_with_labels(labels.as_ref()),
                VectorMatchModifier::Ignoring(labels) => {
                    self.signature_without_labels(labels.as_ref())
                }
            },
        }
    }

    /// Calculate signature for the metric name by the given match modifier without including
    /// the metric group (i.e. only tags are considered).
    pub fn tags_signature_by_match_modifier(
        &self,
        modifier: &Option<VectorMatchModifier>,
    ) -> Signature {
        match modifier {
            None => Signature::from_tags(self),
            Some(m) => match m {
                VectorMatchModifier::On(labels) => self.tags_signature_with_labels(labels.as_ref()),
                VectorMatchModifier::Ignoring(labels) => {
                    self.tags_signature_without_labels(labels.as_ref())
                }
            },
        }
    }
}

impl FromStr for MetricName {
    type Err = RuntimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MetricName::parse(s)
    }
}

impl TryFrom<&str> for MetricName {
    type Error = RuntimeError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        MetricName::parse(s)
    }
}

impl TryFrom<String> for MetricName {
    type Error = RuntimeError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        MetricName::parse(&s)
    }
}

fn count_label_value(
    hm: &mut AHashMap<String, AHashMap<String, usize>>,
    label: &String,
    value: &String,
) {
    let label_counts = hm.entry(label.to_string()).or_default();
    *label_counts.entry(value.to_string()).or_insert(0) += 1;
}

impl Display for MetricName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{{", self.metric_group)?;
        let len = self.tags.len();
        for (i, Tag { key: k, value: v }) in self.tags.iter().enumerate() {
            write!(f, "{}={}", k, enquote('"', v))?;
            if i < len - 1 {
                write!(f, ",")?;
            }
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

        Some(self.tags.len().cmp(&other.tags.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_name() {
        let mut mn = MetricName::default();
        mn.set_metric_group("foo");
        mn.add_tag("bar", "baz");
        mn.add_tag("qux", "quux");
        mn.add_tag("qux", "quuz");
        mn.add_tag("corge", "grault");
        mn.add_tag("garply", "waldo");
        mn.add_tag("fred", "plugh");
        mn.add_tag("xyzzy", "thud");
        mn.add_tag("xyzzy", "thud");
        mn.add_tag("xyzzy", "thud");
        assert_eq!(mn.metric_group, "foo");
        assert_eq!(mn.tags.len(), 6);
        assert_eq!(mn.tags[0].key, "bar");
        assert_eq!(mn.tags[0].value, "baz");

        let mut prev = &mn.tags[0];
        for curr in mn.tags.iter().skip(1) {
            assert!(prev.key < curr.key, "tags are not sorted");
            prev = curr;
        }

        assert_eq!(mn.tag_value("qux"), Some(&String::from("quuz")));
        assert_eq!(mn.tag_value("xyzzy"), Some(&String::from("thud")));
    }

    #[test]
    fn test_add_tag() {
        let mut mn = MetricName::default();
        mn.add_tag("foo", "bar");
        assert_eq!(mn.tags.len(), 1);
        assert_eq!(mn.tags[0].key, "foo");
        assert_eq!(mn.tags[0].value, "bar");

        // replace value if key exists already
        mn.add_tag("foo", "baz");
        assert_eq!(mn.tags.len(), 1);
        assert_eq!(mn.tags[0].key, "foo");
        assert_eq!(mn.tags[0].value, "baz");

        // ensure sort order is maintained
        mn.add_tag("bar", "baz");
        assert_eq!(mn.tags.len(), 2);
        assert_eq!(mn.tags[0].key, "bar");
        assert_eq!(mn.tags[1].key, "foo");
    }

    #[test]
    fn test_duplicate_keys() {
        let mut mn = MetricName::default();
        mn.metric_group = "xxx".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("duplicate", "tag1");
        mn.add_tag("duplicate", "tag2");
        mn.add_tag("tt", "xx");
        mn.add_tag("foo", "abc");
        mn.add_tag("duplicate", "tag3");

        let mut mn_expected = MetricName::default();
        mn_expected.metric_group = "xxx".to_string();
        mn_expected.add_tag("duplicate", "tag3");
        mn_expected.add_tag("foo", "abc");
        mn_expected.add_tag("tt", "xx");

        assert_eq!(mn, mn_expected);
    }

    #[test]
    fn test_remove_tags_on() {
        let mut empty_mn = MetricName::default();
        empty_mn.metric_group = "name".to_string();
        empty_mn.add_tag("key", "value");
        empty_mn.remove_tags_on(&vec![]);
        assert!(
            empty_mn.metric_group.is_empty() && empty_mn.tags.is_empty(),
            "expecting empty metric name got {}",
            &empty_mn
        );

        let mut as_is_mn = MetricName::default();
        as_is_mn.metric_group = "name".to_string();
        as_is_mn.add_tag("key", "value");
        let tags = vec![METRIC_NAME_LABEL.to_string(), "key".to_string()];
        as_is_mn.remove_tags_on(&tags);
        let mut exp_as_is_mn = MetricName::default();
        exp_as_is_mn.metric_group = "name".to_string();
        exp_as_is_mn.add_tag("key", "value");
        assert_eq!(
            exp_as_is_mn, as_is_mn,
            "expecting {} got {}",
            &exp_as_is_mn, &as_is_mn
        );

        let mut mn = MetricName::default();
        mn.metric_group = "name".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        let tags = vec!["baz".to_string()];
        mn.remove_tags_on(&tags);
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_remove_tag() {
        let mut mn = MetricName::default();
        mn.metric_group = "name".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        mn.remove_tag("__name__");
        assert!(
            mn.metric_group.is_empty(),
            "expecting empty metric group name got {}",
            &mn
        );
        mn.remove_tag("foo");
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_remove_tags_ignoring() {
        let mut mn = MetricName::default();
        mn.metric_group = "name".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        let tags = vec![METRIC_NAME_LABEL.to_string(), "foo".to_string()];
        mn.remove_tags_ignoring(&tags);
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_tags_signature_without_labels() {
        let mut mn = MetricName::new("name");
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        let mut exp_mn = MetricName::new("name");
        exp_mn.add_tag("baz", "qux");
        assert_eq!(
            exp_mn.tags_signature_without_labels(&vec!["foo".to_string()]),
            mn.tags_signature_without_labels(&vec!["foo".to_string()]),
            "expecting {} got {}",
            &exp_mn,
            &mn
        );
    }

    #[test]
    fn test_tags_signature_with_labels() {
        let mut mn = MetricName::new("name");
        mn.add_tag("le", "8.799e1");
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        let actual = mn.tags_signature_with_labels(&vec!["baz".to_string()]);
        let expected = exp_mn.tags_signature_with_labels(&vec!["baz".to_string()]);
        assert_eq!(
            actual, expected,
            "expecting {:?} got {:?}",
            expected, actual
        );
    }

    #[test]
    fn test_tags_1() {
        let mut mn = MetricName::new("name");
        mn.add_tag("le", "8.799e1");
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("le", "8.799e1");
        let actual = mn.tags_signature_without_labels(&vec![]);
        let expected = exp_mn.tags_signature_without_labels(&vec![]);
        assert_eq!(
            actual, expected,
            "expecting {:?} got {:?}",
            expected, actual
        );
    }
}
