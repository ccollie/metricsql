use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use crate::time::current_time_millis;

// todo: move to global config
const CACHE_EXPIRE_DURATION: Duration = Duration::from_secs(5 * 60);

fn is_skip_cache(s: &str) -> bool {
    // Skip caching for short strings, since they are usually used only once.
    // This reduces memory usage.
    return s.len() < 16;
}

/// StringTransformCache implements fast transformer for strings.
///
/// It caches string transformation results and returns them back on the next calls
/// without calling the match_func, which may be expensive.
#[derive(Debug)]
pub struct StringTransformCache<R: Clone> {
    inner: Mutex<InnerTransformer<R>>,
    transform_func: fn(s: &str) -> R,
}

#[derive(Clone, Debug)]
struct InnerTransformer<R: Clone> {
    last_cleanup_time: i64,
    map: HashMap<String, CacheEntry<R>>,
}

#[derive(Clone, Debug)]
struct CacheEntry<R: Clone> {
    last_access_time: i64,
    value: R,
}

impl<R: Clone> StringTransformCache<R> {
    /// creates new function which applies transform_func to strings passed to matches()
    ///
    /// match_func must return the same result for the same input.
    pub fn new(transform_func: fn(&str) -> R) -> Self {
        let inner = InnerTransformer {
            last_cleanup_time: current_time_millis(),
            map: HashMap::new(),
        };
        Self {
            inner: Mutex::new(inner),
            transform_func,
        }
    }

    /// Match applies transform to s and returns the result.
    pub fn transform(&self, s: &str) -> R {
        if is_skip_cache(s) {
            return (self.transform_func)(s);
        }

        let ct = current_time_millis();
        let mut inner = self.inner.lock().unwrap();

        if let Some(e) = inner.map.get_mut(s) {
            // Fast path - s match result is found in the cache.
            let val = e.last_access_time + 10;
            if val < ct {
                // Reduce the frequency of e.last_access_time update to once per 10 seconds
                // in order to improve the fast path speed on systems with many CPU cores.
                e.last_access_time = ct;
            }
            return e.value.clone();
        }

        // Slow path - run match_func for s and store the result in the cache.
        let value = (self.transform_func)(s);
        let e = CacheEntry {
            last_access_time: ct,
            value: value.clone(),
        };
        // Make a copy of s in order to limit memory usage to the s length,
        // since the s may point to bigger string.
        // This also protects from the case when s contains unsafe string, which points to a temporary byte slice.
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3227
        inner.map.insert(s.to_string(), e);

        if need_cleanup(&mut inner.last_cleanup_time, ct) {
            // Perform a global cleanup for fsm.m by removing items which weren't accessed during the last 5 minutes.
            let deadline = (ct as u64 - CACHE_EXPIRE_DURATION.as_millis() as u64) as i64;
            inner.map.retain(|_k, v| v.last_access_time >= deadline);
        }

        value
    }
}

impl<R: Clone> Clone for StringTransformCache<R> {
    fn clone(&self) -> Self {
        let inner = self.inner.lock().unwrap();
        Self {
            inner: Mutex::new(inner.clone()),
            transform_func: self.transform_func,
        }
    }
}

fn need_cleanup(last_cleanup_time: &mut i64, current_time: i64) -> bool {
    if *last_cleanup_time + 61_000 >= current_time {
        return false;
    }
    *last_cleanup_time = current_time;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_transform_cache() {
        fn check(s: &str, expected: &str) {
            let stc = StringTransformCache::new(|s| s.to_uppercase());
            for i in 0..10 {
                let result = stc.transform(s);
                assert_eq!(result, expected,
                           "unexpected result for transform({s}) at iteration {i}; got {result}; want {expected}")
            }
        }

        check("a", "A");
        check("a", "A");
        check("b", "B");
        check("", "");
        check("foo", "FOO");
        check("a_b-C", "A_B-C")
    }

    #[test]
    fn test_need_cleanup() {
        fn f(last_cleanup_time: i64, current_time: i64, result_expected: bool) {
            let mut lct = last_cleanup_time;
            let result = need_cleanup(&mut lct, current_time);
            assert_eq!(
                result, result_expected,
                "unexpected result for needCleanup({}, {}); got {}; want {}",
                last_cleanup_time, current_time, result, result_expected
            );
            if result {
                assert_eq!(
                    lct, current_time,
                    "unexpected value for lct; got {}; want current_time={}",
                    lct, current_time
                )
            } else {
                assert_eq!(
                    lct, last_cleanup_time,
                    "unexpected value for lct; got {}; want last_cleanup_time={}",
                    lct, last_cleanup_time
                );
            }
        }
        f(0, 0, false);
        f(0, 61, false);
        f(0, 62, true);
        f(10, 100, true)
    }
}
