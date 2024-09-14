use std::hash::{Hash, Hasher};
use std::ops::Deref;

use metricsql_common::hash::FastHasher;

use crate::types::{MetricName, Label};

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
        let iter = labels.labels.iter();
        Self::with_name_and_labels(&labels.measurement, iter)
    }

    pub fn from_labels(labels: &MetricName) -> Signature {
        let iter = labels.labels.iter();
        Self::with_name_and_labels("", iter)
    }

    pub fn with_name_and_labels<'a>(name: &str, iter: impl Iterator<Item = &'a Label>) -> Self {
        let mut hasher = FastHasher::default();
        let mut has_tags = false;

        if !name.is_empty() {
            hasher.write(name.as_bytes());
        } else {
            hasher.write_u64(EMPTY_NAME_VALUE);
        }
        for tag in iter {
            tag.hash(&mut hasher);
            has_tags = true;
        }
        if !has_tags {
            hasher.write_u64(EMPTY_LIST_SIGNATURE);
        }
        let sig = hasher.finish();
        Signature(sig)
    }
}

impl From<Signature> for u64 {
    fn from(sig: Signature) -> Self {
        sig.0
    }
}

impl From<u64> for Signature {
    fn from(sig: u64) -> Self {
        Signature(sig)
    }
}