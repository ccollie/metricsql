use std::sync::Arc;

use dynamic_lru_cache::DynamicCache;

use crate::regex_util::match_handlers::StringMatchHandler;

const DEFAULT_MAX_SIZE_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_CACHE_SIZE: usize = 100;

#[derive(Clone, Debug)]
pub struct RegexpCacheValue {
    pub(crate) or_values: Vec<String>,
    pub(crate) re_match: StringMatchHandler,
    pub(crate) re_cost: usize,
    pub(crate) literal_suffix: String,
    pub(crate) size_bytes: usize,
}

pub struct RegexpCache {
    max_size_bytes: usize,
    inner: DynamicCache<String, RegexpCacheValue>,
}

impl RegexpCache {
    pub fn new(max_size_bytes: usize) -> Self {
        let cache = DynamicCache::new(DEFAULT_CACHE_SIZE);
        Self {
            max_size_bytes,
            inner: cache,
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<RegexpCacheValue>> {
        let temp = key.to_string();
        self.inner.get(&temp)
    }

    pub fn put(&self, key: &str, value: RegexpCacheValue) -> Arc<RegexpCacheValue> {
        let tmp = key.to_string();
        self.inner.insert(&tmp, value)
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

    pub fn remove(&self, key: &str) -> Option<Arc<RegexpCacheValue>> {
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
