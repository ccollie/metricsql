use std::hash::{Hash, Hasher};
use std::ops::Deref;

use xxhash_rust::xxh3::Xxh3;

use crate::{MetricName, Tag};

#[derive(Debug, Default, Clone, PartialEq, Eq, Copy, Ord, PartialOrd)]
pub struct Signature(u64);

/// implement hash which returns the value of the inner u64
impl Hash for Signature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Deref for Signature {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const EMPTY_LIST_SIGNATURE: u64 = 0x9e3779b97f4a7c15;
const EMPTY_NAME_VALUE: u64 = 0x9e3779b97f4a7c16;

impl Signature {
    pub fn new(labels: &MetricName) -> Self {
        let iter = labels.tags.iter();
        Self::with_name_and_labels(&labels.metric_group, iter)
    }

    pub fn from_tags(labels: &MetricName) -> Signature {
        let iter = labels.tags.iter();
        Self::with_name_and_labels("", iter)
    }

    pub fn with_name_and_labels<'a>(name: &str, iter: impl Iterator<Item = &'a Tag>) -> Self {
        let mut hasher = Xxh3::new();
        let mut has_tags = false;

        if !name.is_empty() {
            hasher.write(name.as_bytes());
        } else {
            hasher.write_u64(EMPTY_NAME_VALUE);
        }
        for tag in iter {
            tag.update_hash(&mut hasher);
            has_tags = true;
        }
        if !has_tags {
            hasher.write_u64(EMPTY_LIST_SIGNATURE);
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
