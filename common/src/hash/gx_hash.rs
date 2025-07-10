use cfg_if::cfg_if;

cfg_if!(
    if #[cfg(all(feature = "gxhash", target_feature="aes"))] {
        use gxhash::{gxhash64, GxHasher, HashMap, HashSet, GxBuildHasher};

        pub type FastHasher = GxHasher;
        pub type FastHashMap<K, V> = HashMap<K, V, GxBuildHasher>;
        pub type FastHashSet<T> = HashSet<T, GxBuildHasher>;

        pub fn fast_hash64(bytes: &[u8]) -> u64 {
            // gxhash is not a cryptographic hash in any case, so a fixed seed
            // is not an issue for our purposes
            gxhash64(bytes, 52731)
        }

        // export HashSetExt for convenience
        pub use gxhash::HashSetExt;
    }
);
