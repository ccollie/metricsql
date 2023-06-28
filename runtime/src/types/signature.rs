use std::hash::Hash;

use xxhash_rust::xxh3::Xxh3;

use crate::{MetricName, Tag};

#[derive(Debug, Default, Clone, PartialEq, Eq, Copy)]
pub struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Signature {
    pub fn new(labels: &MetricName) -> Self {
        let iter = labels.tags.iter();
        Self::from_tag_iter(iter)
    }

    pub fn with_labels(labels: &MetricName, names: &[String]) -> Signature {
        let iter = labels.with_labels_iter(names);
        Self::from_tag_iter(iter)
    }

    /// `signature_without_labels` is just as [`signature`], but only for labels not matching `names`.
    pub fn without_labels(labels: &MetricName, exclude_names: &[String]) -> Signature {
        Self::from_tag_iter(labels.without_labels_iter(exclude_names))
    }

    pub(crate) fn from_tag_iter<'a>(iter: impl Iterator<Item=&'a Tag>) -> Self {
        let mut hasher = Xxh3::new();
        for tag in iter {
            tag.update_hash(&mut hasher);
        }
        let sig = hasher.digest();
        Signature(sig)
    }
}

impl From<Signature> for u64 {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}
