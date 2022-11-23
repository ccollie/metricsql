use std::collections::HashMap;
use std::hash::Hasher;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use once_cell::sync::Lazy;
/// import commonly used items from the prelude:
use rand::prelude::*;
use xxhash_rust::xxh3::Xxh3;

use lib::{
    compress_lz4,
    decompress_lz4,
    get_pooled_buffer,
    marshal_fixed_int,
    marshal_var_int,
    unmarshal_fixed_int,
    unmarshal_var_int
};
use metricsql::ast::{Expression, LabelFilter};

use crate::{EvalConfig, marshal_timeseries_fast, Timeseries};
use crate::cache::default_result_cache_storage::DefaultResultCacheStorage;
use crate::cache::traits::RollupResultCacheStorage;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::{copy_timeseries_shallow, unmarshal_timeseries_fast};
use crate::traits::{Timestamp, TimestampTrait};
use crate::utils::{memory_limit, MemoryLimiter};

/// The maximum duration since the current time for response data, which is always queried from the
/// original raw data, without using the response cache. Increase this value if you see gaps in responses
/// due to time synchronization issues between this library and data sources. See also
/// -search.disableAutoCacheReset
/// TODO: move to EvalConfig
static CACHE_TIMESTAMP_OFFSET: i64 = 5000;

static ROLLUP_RESULT_CACHE_KEY_PREFIX: Lazy<u64> = Lazy::new(|| random::<u64>() );


fn get_default_cache_size() -> u64 {
    // todo: tune this
    let mut n = memory_limit().unwrap() / 16;
    if n <= 0 {
        n = 1024 * 1024;
    }
    n
}

struct Inner {
    cache: Box<dyn RollupResultCacheStorage + Send + Sync>,
    hasher: Xxh3
}

pub struct RollupResultCache {
    inner: Mutex<Inner>,
    memory_limiter: MemoryLimiter,
    max_marshaled_size: u64,
    cache_key_suffix: AtomicU64,
}

impl Default for RollupResultCache {
    fn default() -> Self {
        let size = get_default_cache_size();
        Self::with_size(size as usize)
    }
}

impl RollupResultCache {
    // todo: pass in cache

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_size(max_size: usize) -> Self {
        let mut rng = thread_rng();

        let cache = Box::new(DefaultResultCacheStorage::new(max_size));
        let hasher = Xxh3::default();
        let suffix: u64 = rng.gen_range((1 << 16)..(1 << 31));
        let memory_limiter = MemoryLimiter::new(max_size);

        let inner = Inner {
            cache,
            hasher
        };

        Self {
            inner: Mutex::new(inner),
            memory_limiter,
            max_marshaled_size: (max_size as u64 / 4_u64),
            cache_key_suffix: AtomicU64::new( suffix ),
        }
    }

    pub fn reserve_memory(&self, size: usize) -> bool {
        self.memory_limiter.get(size)
    }

    pub fn release_memory(&self, size: usize) -> RuntimeResult<()> {
        self.memory_limiter.put(size)
    }

    pub fn memory_limit(&self) -> usize {
        self.memory_limiter.max_size
    }

    pub fn get(&self, ec: &EvalConfig, expr: &Expression, window: i64) -> RuntimeResult<(Option<Vec<Timeseries>>, i64)> {
        if !ec.may_cache() {
            return Ok((None, ec.start));
        }

        // Obtain tss from the cache.
        let mut meta_info_buf = get_pooled_buffer(1024);

        let mut inner = self.inner.lock().unwrap();

        let hash = marshal_rollup_result_cache_key(&mut inner.hasher,
                                        expr, window, ec.step, &ec.enforced_tag_filterss);

        if !inner.cache.get(&hash.to_ne_bytes(), &mut meta_info_buf) || meta_info_buf.len() == 0 {
            return Ok((None, ec.start));
        }
        let mut mi: RollupResultCacheMetaInfo;

        match RollupResultCacheMetaInfo::from_buf(meta_info_buf.as_slice()) {
            Err(err) => {
                let msg = format!("BUG: cannot unmarshal RollupResultCacheMetainfo; {:?}", err);
                return Err(RuntimeError::SerializationError(msg));
            },
            Ok(m) => mi = m
        }

        let key = mi.get_best_key(ec.start, ec.end)?;
        if key.prefix == 0 && key.suffix == 0 {
            return Ok((None, ec.start));
        }
        
        let mut bb = get_pooled_buffer(2048);
        key.marshal(&mut bb);

        let mut compressed_result_buf = get_pooled_buffer(2048);

        if !inner.cache.get_big(bb.as_slice(), &mut compressed_result_buf) ||
            compressed_result_buf.len() == 0 {
            mi.remove_key(key);
            mi.marshal(&mut meta_info_buf);
            let hash = marshal_rollup_result_cache_key(&mut inner.hasher, expr,
                    window, ec.step, &ec.enforced_tag_filterss);

            inner.cache.set(&hash.to_ne_bytes(), meta_info_buf.as_slice());
            return Ok((None, ec.start));
        }
        // we don't need cache past this point
        drop(inner);

        // Decompress into newly allocated byte slice, since tss returned from unmarshalTimeseriesFast
        // refers to the byte slice, so it cannot be returned to the resultBufPool.

        let mut tss = match decompress_lz4(&compressed_result_buf) {
            Ok(uncompressed) => {
                match unmarshal_timeseries_fast(&uncompressed) {
                    Ok(tss) => tss,
                    Err(_) => {
                        let msg = format!("BUG: cannot unmarshal timeseries from RollupResultCache:; it looks like it was improperly saved");
                        return Err(RuntimeError::from(msg))
                    }
                }
            },
            Err(err) => {
                return Err(RuntimeError::from(
                    format!("BUG: cannot decompress resultBuf from RollupResultCache: {:?}; it looks like it was improperly saved", err)
                ));
            }
        };

        // Extract values for the matching timestamps
        let mut i = 0;
        let mut j;
        {
            let timestamps = &tss[0].timestamps;
            while i < timestamps.len() && timestamps[i] < ec.start {
                i += 1
            }
            if i == timestamps.len() {
                // no matches.
                return Ok((None, ec.start));
            }
            if timestamps[i] != ec.start {
                // The cached range doesn't cover the requested range.
                return Ok((None, ec.start));
            }

            j = timestamps.len() - 1;
            while j > 0 && timestamps[j] > ec.end {
                j -= 1;
            };
            if j <= i {
                // no matches.
                return Ok((None, ec.start));
            }
        }

        for ts in tss.iter_mut() {
            let ts_slice = &ts.timestamps[i..j];
            let value_slice = &ts.values[i..j];
            ts.timestamps = Arc::new(ts_slice.to_vec());
            ts.values = value_slice.to_vec();
        }

        let timestamps = &tss[0].timestamps;
        let new_start = timestamps[timestamps.len() - 1] + ec.step;
        return Ok((Some(tss), new_start));
    }

    pub fn put(&self, ec: &EvalConfig, expr: &Expression, window: i64, tss: &Vec<Timeseries>) -> RuntimeResult<()> {
        if tss.len() == 0 || !ec.may_cache() {
            return Ok(());
        }

        // Remove values up to currentTime - step - CACHE_TIMESTAMP_OFFSET,
        // since these values may be added later.
        let timestamps = &tss[0].timestamps;
        let deadline = (Timestamp::now() as f64 / 1e6_f64) as i64 - ec.step - CACHE_TIMESTAMP_OFFSET;
        let mut i = timestamps.len() - 1;
        while i > 0 && timestamps[i] > deadline {
            i -= 1;
        }
        if i == 0 {
            // Nothing to store in the cache.
            return Ok(());
        }

        let start = timestamps[0];
        let mut end = timestamps[timestamps.len() - 1];

        // shadow variable because of the following:
        let mut tss = tss;
        let rvs = tss;
        if i < timestamps.len() {
            end = timestamps[i];
            let ts_slice = tss[0].timestamps[0..i].to_vec();
            let new_stamps = Arc::new(ts_slice);
            // Make a copy of tss and remove unfit values
            let mut _rvs = copy_timeseries_shallow(tss);
            for ts in _rvs.iter_mut() {
                ts.timestamps = Arc::clone(&new_stamps);
                ts.values.resize(i, 0.0);
            }
            tss = rvs
        }

        // Store tss in the cache.
        let mut metainfo_key = get_pooled_buffer(512);
        let mut metainfo_buf = get_pooled_buffer(1024);

        let mut inner = self.inner.lock().unwrap();

        let hash = marshal_rollup_result_cache_key(&mut inner.hasher, expr, window, ec.step,
                                        &ec.enforced_tag_filterss);

        let found = inner.cache.get(&hash.to_ne_bytes(), &mut metainfo_buf);
        let mut mi = if found && metainfo_buf.len() > 0 {
            match RollupResultCacheMetaInfo::from_buf(metainfo_buf.deref_mut()) {
                Err(_) => {
                    let msg = "BUG: cannot unmarshal RollupResultCacheMetainfo; it looks like it was improperly saved";
                    return Err(RuntimeError::SerializationError(msg.to_string()));
                },
                Ok(mi) => mi
            }
        } else {
          RollupResultCacheMetaInfo::new()
        };

        if mi.covers_time_range(start, end) {
            // qt.Printf("series on the given timeRange=[{}..{}] already exist in the cache", start, end);
            return Ok(());
        }

        let mut result_buf = get_pooled_buffer(2048);
        // should we handle error here and consider it a cache miss ?
        marshal_timeseries_fast(result_buf.deref_mut(), tss, self.max_marshaled_size as usize, ec.step)?;
        if result_buf.len() == 0 {
            return Ok(());
        }

        let compressed_buf = compress_lz4(&result_buf);

        let suffix = self.cache_key_suffix.fetch_add(1, Ordering::Relaxed);
        let key = RollupResultCacheKey::new(suffix);

        metainfo_key.clear();
        key.marshal(metainfo_key.deref_mut());

        inner.cache.set_big(&metainfo_key, compressed_buf.as_slice());

        mi.add_key(key, start, end)?;
        mi.marshal(&mut metainfo_buf);
        inner.cache.set(metainfo_key.as_slice(), metainfo_buf.as_slice());
        return Ok(())
    }
}

// var resultBufPool bytesutil.ByteBufferPool

/// Increment this value every time the format of the cache changes.
const ROLLUP_RESULT_CACHE_VERSION: u8 = 8;

fn marshal_rollup_result_cache_key(
    hasher: &mut Xxh3,
    expr: &Expression,
    window: i64,
    step: i64,
    etfs: &Vec<Vec<LabelFilter>>) -> u64 {

    hasher.reset();
    
    let prefix: u64 = *ROLLUP_RESULT_CACHE_KEY_PREFIX.deref();
    hasher.write_u64(prefix);
    hasher.write_u8(ROLLUP_RESULT_CACHE_VERSION);
    hasher.write_i64(window);
    hasher.write_i64(step);
    hasher.write(format!("{}", expr).as_bytes());

    for etf in etfs.iter() {
        for f in etf {
            hasher.write(f.as_string().as_bytes())
        }
    }

    hasher.digest()
}

/// merge_timeseries concatenates b with a and returns the result.
///
/// Preconditions:
/// - a mustn't intersect with b.
/// - a timestamps must be smaller than b timestamps.
///
/// Postconditions:
/// - a and b cannot be used after returning from the call.
pub fn merge_timeseries(a: Vec<Timeseries>, b: Vec<Timeseries>, b_start: i64, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
    let shared_timestamps = ec.timestamps();
    if b_start == ec.start {
        // Nothing to merge - b covers all the time range.
        // Verify b is correct.
        let mut second = b;
        for ts_second in second.iter_mut() {
            ts_second.timestamps = Arc::clone(&shared_timestamps);
            validate_timeseries_length(&ts_second)?;
        }
        // todo(perf): if this clone the most efficient
        return Ok( second );
    }

    let mut map: HashMap<String, Timeseries> = HashMap::with_capacity(a.len());

    for ts in a.into_iter() {
        let key = ts.metric_name.to_string();
        map.insert(key, ts);
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(map.len());

    for mut ts_second in b.into_iter() {
        let key = ts_second.metric_name.to_string();

        let mut tmp: Timeseries = Timeseries {
            metric_name: std::mem::take(&mut ts_second.metric_name), // todo(perf): how to avoid clone() (use into)?
            timestamps: Arc::clone(&shared_timestamps),
            values: Vec::with_capacity(shared_timestamps.len())
        };

        match map.get_mut(&key) {
            None => {
                let mut t_start = ec.start;
                while t_start < b_start {
                    tmp.values.push(f64::NAN);
                    t_start += ec.step;
                }
            },
            Some(ts_a) => {
                tmp.values.extend_from_slice(&ts_a.values);
                map.remove(&key);
            }
        }

        tmp.values.append(&mut ts_second.values);
        validate_timeseries_length(&ts_second)?;

        rvs.push(tmp);
    }

    // todo: collect() then rvs.extend()
    // Copy the remaining timeseries from m.
    for ts_a in map.values_mut() {
        let mut t_start = b_start;
        while t_start <= ec.end {
            ts_a.values.push(f64::NAN);
            t_start += ec.step;
        }

        validate_timeseries_length(&ts_a)?;

        rvs.push(std::mem::take(ts_a));
    }

    Ok(rvs)
}

fn validate_timeseries_length(ts: &Timeseries) -> RuntimeResult<()> {
    if ts.values.len() != ts.timestamps.len() {
        let msg = format!("mismatched timestamp/value length in timeseries; got {}; want {}",
               ts.values.len(), ts.timestamps.len());
        return Err( RuntimeError::InvalidState(msg) )
    }
    Ok(())
}

struct RollupResultCacheMetaInfo {
    entries: Vec<RollupResultCacheMetaInfoEntry>,
}

impl RollupResultCacheMetaInfo {
    fn new() -> Self {
        Self {
            entries: vec![]
        }
    }

    fn from_buf(buf: &[u8]) -> RuntimeResult<Self> {
        let (res, _) = Self::unmarshal(buf)?;
        Ok(res)
    }

    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_fixed_int::<usize>(dst, self.entries.len());
        for entry in &self.entries {
            entry.marshal(dst);
        }
    }

    fn unmarshal(buf: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfo, &[u8])> {
        let mut src = buf;

        let entries_len: usize;
        match unmarshal_fixed_int::<usize>(src) {
            Ok((v, tail)) => {
                entries_len = v;
                src = tail;
            },
            Err(_) => {
                let msg = format!("cannot unmarshal len(entries) from {} bytes; need at least {} bytes", src.len(), 4);
                return Err(RuntimeError::SerializationError(msg));
            }
        }

        let mut entries: Vec<RollupResultCacheMetaInfoEntry> = Vec::with_capacity(entries_len);
        let mut i = 0;
        while i < entries_len {
            match RollupResultCacheMetaInfoEntry::read(src) {
                Ok((v, tail)) => {
                    entries.push(v);
                    src = tail;
                },
                Err(err) => {
                    return Err(RuntimeError::from(format!("cannot unmarshal entry #{}: {:?}", i, err)));
                }
            }
            i += 1;
        }

        if i < entries_len {
            return Err(RuntimeError::from(format!("expected {} cache entries: got {}", entries_len, entries.len())));
        }

        if src.len() > 0 {
            return Err(RuntimeError::from(format!("unexpected non-empty tail left; len(tail)={}", src.len())));
        }

        Ok(
            (Self { entries }, src)
        )
    }

    fn covers_time_range(&self, start: i64, end: i64) -> bool {
        if start > end {
            // todo: remove panic. return Result instead
            panic!("BUG: start cannot exceed end; got {} vs {}", start, end)
        }
        for entry in self.entries.iter() {
            if start >= entry.start && end <= entry.end {
                return true
            }
        }
        return false
    }

    fn get_best_key(&self, start: i64, end: i64) -> RuntimeResult<RollupResultCacheKey> {
        if start > end {
            return Err(
                RuntimeError::ArgumentError(
                    format!("BUG: start cannot exceed end; got {} vs {}", start, end)
                )
            )
        }
        let mut best_key: RollupResultCacheKey = RollupResultCacheKey::default();
        let mut d_max: i64 = 0;
        for e in self.entries.iter() {
            if start < e.start {
                continue;
            }
            let mut d = e.end - start;
            if end <= e.end {
                d = end - start
            }
            if d >= d_max {
                d_max = d;
                best_key = e.key;
            }
        }
        Ok(best_key)
    }

    fn add_key(&mut self, key: RollupResultCacheKey, start: i64, end: i64) -> RuntimeResult<()> {
        if start > end {
            // todo: return Result
            return Err(RuntimeError::ArgumentError(
                format!("BUG: start cannot exceed end; got {} vs {}", start, end))
            );
        }

        self.entries.push(RollupResultCacheMetaInfoEntry {
            start,
            end,
            key,
        });

        if self.entries.len() > 30 {
            // Remove old entries.
            self.entries.drain(0..9);
        }

        Ok(())
    }

    fn remove_key(&mut self, key: RollupResultCacheKey) {
        self.entries.retain(|x| x.key != key)
    }
}

impl Default for RollupResultCacheMetaInfo {
    fn default() -> Self {
        Self {
            entries: vec![]
        }
    }
}

#[derive(Default, Clone, Eq, Hash)]
pub(self) struct RollupResultCacheMetaInfoEntry {
    start: i64,
    end: i64,
    key: RollupResultCacheKey,
}

impl RollupResultCacheMetaInfoEntry {

    fn read(src: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfoEntry, &[u8])> {
        Self::unmarshal(src)
    }

    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_var_int(dst, self.start);
        marshal_var_int(dst, self.end);
        self.key.marshal(dst);
    }

    fn unmarshal(src: &[u8]) -> RuntimeResult<(Self, &[u8])> {

        if src.len() < 8 {
            return Err(RuntimeError::SerializationError(
                format!("cannot unmarshal start from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }
        
        let mut src = src;
        let mut res = Self::default();

        match unmarshal_var_int::<i64>(src) {
            Err(err) => {
                return Err(RuntimeError::SerializationError(
                    format!("cannot unmarshal start: {:?}", err)
                ));
            },
            Ok((start, tail)) => {
                res.start = start;
                src = tail;
            }
        }
        
        match unmarshal_var_int::<i64>(src) {
            Err(err) => {
                return Err(RuntimeError::SerializationError(
                    format!("cannot unmarshal end: {:?}", err)
                ));
            },
            Ok((start, tail)) => {
                res.end = start;
                src = tail;
            }
        }

        (res.key, src) = RollupResultCacheKey::unmarshal(src)?;

        Ok((res, src))
    }
}

impl PartialEq for RollupResultCacheMetaInfoEntry {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start &&
            self.end == other.end &&
            self.key == other.key
    }
}

/// RollupResultCacheKey must be globally unique across nodes,
/// so it has prefix and suffix.
#[derive(Hash, Copy, Eq, PartialEq, Clone)]
pub struct RollupResultCacheKey {
    prefix: u64,
    suffix: u64,
}

impl Default for RollupResultCacheKey {
    fn default() -> Self {
        Self::new(0)
    }
}

impl RollupResultCacheKey {
    fn new(suffix: u64) -> Self {
        // not sure if this is safe
        RollupResultCacheKey {
            prefix: *ROLLUP_RESULT_CACHE_KEY_PREFIX.deref(),
            suffix
        }
    }

    fn marshal(&self, dst: &mut Vec<u8>)  {
        dst.push(ROLLUP_RESULT_CACHE_VERSION);
        marshal_var_int(dst, self.prefix);
        marshal_var_int(dst, self.suffix);
    }

    pub(self) fn unmarshal(src: &[u8]) -> RuntimeResult<(RollupResultCacheKey, &[u8])> {
        if src.len() < 8 {
            return Err(RuntimeError::SerializationError(
                format!("cannot unmarshal key prefix from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }
        let mut cursor: &[u8];
        let prefix: u64;

        match unmarshal_var_int::<u64>(src) {
            Err(_) => {
                return Err(RuntimeError::SerializationError("cannot unmarshal prefix".to_string()));
            },
            Ok((val, tail)) => {
                prefix = val;
                cursor = tail;
            }
        }

        let suffix: u64;

        if src.len() < 8 {
            return Err(RuntimeError::from(
                format!("cannot unmarshal key suffix from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }

        match unmarshal_var_int::<u64>(cursor) {
            Err(err) => {
                let msg = format!("error unmarshalling suffix: {:?}", err);
                return Err(RuntimeError::SerializationError(msg));
            },
            Ok((val, tail)) => {
                suffix = val;
                cursor = tail;
            }
        }

       Ok((RollupResultCacheKey{ prefix, suffix }, cursor))
    }
}