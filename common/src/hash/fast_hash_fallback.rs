use ahash::{AHashMap, AHashSet, AHasher};
use std::hash::Hasher;

pub type FastHasher = AHasher;
pub type FastHashMap<K, V> = AHashMap<K, V>;
pub type FastHashSet<T> = AHashSet<T>;

pub fn fast_hash64(bytes: &[u8]) -> u64 {
    let mut hasher = FastHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

pub use ahash::HashSetExt;
