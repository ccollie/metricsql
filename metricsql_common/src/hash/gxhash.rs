use gxhash::{gxhash64, GxHasher, GxHashMap, GxHashSet};

pub type FastHasher = GxHasher;
pub type FastHashMap<K, V> = GxHashMap<K, V>;
pub type FastHashSet<T> = GxHashSet<T>;

pub fn fast_hash64(bytes: &[u8]) -> u64 {
    // gxhash is not a cryptographic hash in any case so a fixed seed
    // is not an issue for our purposes
    gxhash64(bytes, 52731)
}
