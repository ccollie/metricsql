use std::sync::Arc;

use dynamic_lru_cache::DynamicCache;

#[derive(Clone, Default, Debug)]
pub(crate) struct PrefixSuffix {
    pub(crate) prefix: String,
    pub(crate) suffix: String,
}

impl PrefixSuffix {
    pub fn new(prefix: String, suffix: String) -> Self {
        Self { prefix, suffix }
    }

    pub fn size_bytes(&self) -> usize {
        &self.prefix.len() + &self.suffix.len() + std::mem::size_of::<Self>()
    }
}

pub struct PrefixCache {
    max_size_bytes: usize,
    size_bytes: usize,
    inner: DynamicCache<String, PrefixSuffix>,
}

impl PrefixCache {
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            inner: DynamicCache::new(100),
            size_bytes: 0,
            max_size_bytes,
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<Arc<PrefixSuffix>> {
        let key_ = key.to_string();
        self.inner.get(&key_)
    }

    pub(crate) fn put(&self, key: &str, value: PrefixSuffix) -> Arc<PrefixSuffix> {
        let key_ = key.to_string();
        let size = value.size_bytes();
        if self.inner.mem_len() + size > self.max_size_bytes {
            // todo
        }
        //self.inner.size_bytes += size;
        self.inner.insert(&key_, value)
    }

    /// returns the number of cached regexps for tag filters.
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.inner.clear_cache();
    }

    pub fn remove(&self, key: &str) -> Option<Arc<PrefixSuffix>> {
        let temp_value = key.to_string();
        self.inner.pop(&temp_value)
    }

    pub fn misses(&self) -> u64 {
        let (_, m) = self.inner.hits_misses();
        m
    }

    pub fn requests(&self) -> u64 {
        let (hits, misses) = self.inner.hits_misses();
        hits + misses
    }

    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }
}
