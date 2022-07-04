use regex::{Error, Regex};
use std::{
    sync::{Arc, Mutex},
};
use std::collections::HashMap;
use once_cell::sync::OnceCell;

// REGEXP_CACHE_CHARS_MAX limits the max number of chars stored in regexp cache across all entries.
//
// We limit by number of chars since calculating the exact size of each regexp is problematic,
// while using chars seems like universal approach for short and long regexps.
const REGEXP_CACHE_CHARS_MAX: usize = 1e6 as usize;


fn get_cache() -> &'static RegexpCache {
	static INSTANCE: OnceCell<RegexpCache> = OnceCell::new();
	INSTANCE.get_or_init(|| {
		/*
			metrics.NewGauge(`vm_cache_requests_total{type="promql/regexp"}`, || {
				return float64(rc.Requests())
			})
	metrics.NewGauge(`vm_cache_misses_total{type="promql/regexp"}`, || {
		return float64(rc.Misses())
	})
	metrics.NewGauge(`vm_cache_entries{type="promql/regexp"}`, || {
		return float64(rc.Len())
	})
	metrics.NewGauge(`vm_cache_chars_current{type="promql/regexp"}`, || {
		return float64(rc.CharsCurrent())
	})
	metrics.NewGauge(`vm_cache_chars_max{type="promql/regexp"}`, || {
		return float64(rc.chars_limit)
	})


		 */
		RegexpCache::new(REGEXP_CACHE_CHARS_MAX)
	})
}


// CompileRegexpAnchored returns compiled regexp `^re$`.
pub fn compile_regexp_anchored(re: &str) -> Result<&Regex, Error> {
	let re_anchored = format!("^(?:{})$", re);
	return compile_regexp(&re_anchored)
}

// CompileRegexp returns compile regexp re.
pub fn compile_regexp(re: &str) -> Result<&Regex, Error> {
	match Some(rcv) = get_cache().get(re) {
		Ok(v) => {
			if v.r.is_some() {
				Ok(v.r.unwrap())
			}
			return Err(v.err)
		}
		None(..) => {
			match Regex::new(re) {
				Ok(r) => {
					let rcv = RegexpCacheValue {
						r: Some(r),
						err: None
					};
					get_cache().put(re, rcv);
					Ok(&r)
				}
				Err(err) => {
					let rcv = RegexpCacheValue {
						err: Some(err),
						r: None
					};
					get_cache().put(re, rcv);
					Err(err.clone())
				}
			}
		}
	}
}

struct RegexpCacheValue {
	r: Some<Regex>,
	err: Some<Error>
}

struct RegexpCacheInner {
	// Move atomic counters to the top of struct for 8-byte alignment on 32-bit arch.
	// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/212
	requests: u64,
	misses:   u64,

	// charsCurrent stores the total number of characters used in stored regexps.
	// is used for memory usage estimation.
	chars_current: usize,

	// charsLimit is the maximum number of chars the regexpCache can store.
	chars_limit: usize,

	m: HashMap<String, RegexpCacheValue>
}

impl RegexpCacheInner {

    pub fn get(&mut self, regexp: &str) -> Some(RegexpCacheValue) {
        self.requests = self.requests + 1;
        rcv = rc.m.get(regexp);
        if rcv.is_none() {
            self.misses = self.misses + 1;
        }
        rcv
    }

    pub fn put(&mut self, regexp: &str, rcv: RegexpCacheValue) {
    	if self.chars_current > self.chars_limit {
    		// Remove items accounting for 10% chars from the cache.
    		let mut overflow = (self.chars_limit as f64 * 0.1);
    		for (k, v) in self.m {
    			self.m.remove(&k);

    			let size = k.len();
    			overflow = overFlow - size;
    			self.chars_current -= size;

    			if overflow <= 0.0 {
    				break
    			}
    		}
    	}
    	self.m.set(regexp, rcv);
    	self.chars_current = self.chars_current + regexp.len();
    }
}

pub(crate) struct RegexpCache {
	inner: Arc<Mutex<RegexpCacheInner>>
}

impl RegexpCache {
    pub fn new(chars_limit: usize) -> Self {
        let cache = RegexpCacheInner {
          m: HashMap::new(),
          chars_limit,
          chars_current: 0,
          requests: 0,
          misses: 0
        };

        RegexpCache {
            inner: Arc::new(Mutex::new(cache))
        }
    }

    #[inline]
    pub fn misses(self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.misses
    }

    #[inline]
    pub fn requests(self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.requests
    }

    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.m.len()
    }

    pub fn chars_current(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.chars_current
    }

    pub(crate) fn put(&mut self, regexp: &str, rcv: RegexpCacheValue) {
    	let mut inner = self.inner.lock().unwrap();
    	inner.put(str, rcv);
    }

    pub fn get(&self, regexp: &str) -> Some(RegexpCacheValue) {
		let mut inner = self.inner.lock().unwrap();
		inner.get(regexp)
    }
}
