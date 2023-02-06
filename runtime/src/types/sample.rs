use std::hash::{Hasher};
use xxhash_rust::xxh3::Xxh3;
use super::{MetricName, Timestamp};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Sample {
    pub metric: MetricName,
    pub timestamp: Timestamp,
    pub value: f64
}

impl Sample {
    pub fn get_hash(&self) -> u64 {
        let mut hasher: Xxh3 = Xxh3::new();
        hasher.write_u64(self.metric.fast_hash());
        hasher.write_i64(self.timestamp);
        hasher.write_u64(self.value.to_bits());
        hasher.finish()
    }
}