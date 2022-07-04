use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;
use lib::error::Error;
use metricsql::types::Expression;

const PARSE_CACHE_MAX_LEN: usize = 1000;

struct ParseCacheInner {
    requests: u64,
    misses: u64,
    m: HashMap<String, ParseCacheValue>
}

pub (crate) struct ParseCacheValue {
    pub(crate) expr: Option<Expression>,
    pub(crate) err: Option<Error>
}

pub struct ParseCache {
    inner: Arc<Mutex<ParseCacheInner>>
}

impl ParseCache {
    pub fn new() -> Self {
        let inner = ParseCacheInner {
            requests: 0,
            misses: 0,
            m: HashMap::new()
        };
        ParseCache {
            inner: Arc::new(Mutex::new(inner))
        }
    }

    pub fn len(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.m.len()
    }

    pub fn misses(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.misses
    }

    pub fn requests(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.requests
    }

    pub fn get(&mut self, key: String) -> ParseCacheValue {
       let inner = self.inner.lock().unwrap();
       inner.requests = inner.requests + 1;
       let entry = inner.get(key);
       if entry.is_none() {
           inner.misses = inner.misses + 1;
       }
       return entry;
    }

    pub fn put(&mut self, q: String, pcv: ParseCacheValue) {
        let inner = self.inner.lock().unwrap();
        let mut overflow = inner.m.len() - PARSE_CACHE_MAX_LEN;
        if overflow > 0 {
            // Remove 10% of items from the cache.
            overflow = ((lpc.m.len() as f64) * 0.1) as usize;
            for k in inner.m.keys() {
                inner.m.remove(k);
                overflow = overflow - 1;
                if overflow <= 0 {
                    break
                }
            }
        }
        inner.m.insert(q, pcv);
    }
}

pub(crate) fn get_parse_cache() -> &'static ParseCache {
    static INSTANCE: OnceCell<ParseCache> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        /*
	metrics.NewGauge(`vm_cache_requests_total{type="promql/parse"}`, func() float64 {
		return float64(pc.Requests())
	})
	metrics.NewGauge(`vm_cache_misses_total{type="promql/parse"}`, func() float64 {
		return float64(pc.Misses())
	})
	metrics.NewGauge(`vm_cache_entries{type="promql/parse"}`, func() float64 {
		return float64(pc.Len())
	})
         */
        ParseCache::new()
    })
}