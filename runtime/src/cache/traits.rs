
/// Rollup Result Cache operations
pub trait RollupResultCacheStorage {
    /// Attempt to retrieve a cached value
    fn get(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool;

    /// Searches for the value for the given k, appends it to dst
    /// and returns the result.
    ///
    /// Returns only values stored via set_big. It doesn't work
    /// with values stored via other methods.
    ///
    /// k contents may be modified after returning from GetBig.
    fn get_big(&mut self, k: &[u8], dst: &mut Vec<u8>) -> bool;

    /// Insert a key, value pair
    fn set(&mut self, k: &[u8], v: &[u8]);

    /// sets (k, v) to c where v.len() may exceed 64KB.
    ///
    /// get_big must be used for reading stored values.
    ///
    /// The stored entry may be evicted at any time either due to cache
    /// overflow or due to unlikely hash collision.
    /// Pass higher maxBytes value to New if the added items disappear
    /// frequently.
    ///
    /// It is safe to store entries smaller than 64KB with SetBig.
    ///
    /// k and v contents may be modified after returning from SetBig.
    fn set_big(&mut self, k: &[u8], v: &[u8]);

    /// Remove all cached values. Keeps the allocated memory for reuse.
    fn clear(&mut self);

    /// Remove all cached values. Free memory and return to initial state
    fn reset(&mut self);

    /// Reset misses/hits counters
    fn reset_metrics(&mut self) {}

    /// Return the current cache size (number of elements)
    fn len(&self) -> usize;

    /// Return the number of times a cached value was successfully retrieved
    fn hits(&self) -> Option<u64> {
        None
    }

    /// Return the number of times a cached value was unable to be retrieved
    fn misses(&self) -> Option<u64> {
        None
    }
}