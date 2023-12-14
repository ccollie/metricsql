use ahash::{AHasher, AHashMap, AHashSet, RandomState};

pub type FastHasher = AHasher;
pub type FastHashMap<K, V> = AHashMap<K, V>;
pub type FastHashSet<T> = AHashSet<T>;

pub fn fast_hash64(bytes: &[u8]) -> u64 {
    let build_hasher = RandomState::default();
    build_hasher.hash_one(b)
}
