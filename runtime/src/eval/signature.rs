use crate::MetricName;
use metricsql::common::GroupModifier;
use std::hash::Hash;
use xxhash_rust::xxh3::Xxh3;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Signature {
    pub fn with_group_modifier(
        labels: &MetricName,
        group_modifier: &Option<GroupModifier>,
    ) -> Signature {
        let mut hasher = Xxh3::new();
        let sig = labels.get_hash_by_group_modifier(&mut hasher, group_modifier);
        Signature(sig)
    }

    pub fn with_labels(labels: &MetricName, names: &[String]) -> Signature {
        let mut hasher = Xxh3::new();
        let sig = labels.hash_with_labels(&mut hasher, names);
        Signature(sig)
    }

    /// `signature_without_labels` is just as [`signature`], but only for labels not matching `names`.
    pub fn without_labels(labels: &MetricName, exclude_names: &[String]) -> Signature {
        let mut hasher = Xxh3::new();
        let sig = labels.hash_without_labels(&mut hasher, exclude_names);
        Signature(sig)
    }
}

impl From<Signature> for u64 {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}
