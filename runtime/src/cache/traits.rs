
/// Rollup Result Cache operations
pub trait RollupResultCacheProvider {
    /// Attempt to retrieve a cached value
    fn cache_get(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool;

    /// Searches for the value for the given k, appends it to dst
    /// and returns the result.
    ///
    /// Returns only values stored via SetBig. It doesn't work
    /// with values stored via other methods.
    ///
    /// k contents may be modified after returning from GetBig.
    fn cache_get_big(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool;

    /// Insert a key, value pair
    fn cache_set(&mut self, k: &[u8], v: &[u8]);

    /// cache_set_big sets (k, v) to c where v.len() may exceed 64KB.
    ///
    /// cache_get_big must be used for reading stored values.
    ///
    /// The stored entry may be evicted at any time either due to cache
    /// overflow or due to unlikely hash collision.
    /// Pass higher maxBytes value to New if the added items disappear
    /// frequently.
    ///
    /// It is safe to store entries smaller than 64KB with SetBig.
    ///
    /// k and v contents may be modified after returning from SetBig.
    fn cache_set_big(&mut self, k: &[u8], v: &[u8]);

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
}