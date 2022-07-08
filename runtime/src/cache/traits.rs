
/// Cache operations
pub trait RollupResultCacheProvider<K, V> {
    /// Attempt to retrieve a cached value
    fn cache_get(&mut self, k: &K) -> Option<&V>;

    /// Attempt to retrieve a cached value with mutable access
    fn cache_get_mut(&mut self, k: &K) -> Option<&mut V>;

    /// Insert a key, value pair and return the previous value
    fn cache_set(&mut self, k: K, v: V) -> Option<V>;

    /// Remove a cached value
    fn cache_remove(&mut self, k: &u8) -> Option<V>;

    /// Remove all cached values. Keeps the allocated memory for reuse.
    fn cache_clear(&mut self);

    /// Remove all cached values. Free memory and return to initial state
    fn cache_reset(&mut self);

    /// Reset misses/hits counters
    fn cache_reset_metrics(&mut self) {}

    /// Return the current cache size (number of elements)
    fn cache_size(&self) -> usize;

    /// Return the number of times a cached value was successfully retrieved
    fn cache_hits(&self) -> Option<u64> {
        None
    }

    /// Return the number of times a cached value was unable to be retrieved
    fn cache_misses(&self) -> Option<u64> {
        None
    }

    /// Set the lifespan of cached values, returns the old value
    fn cache_set_lifespan(&mut self, _seconds: u64) -> Option<u64> {
        None
    }
}