use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use chrono::Duration;

use effective_limits::memory_limit;
use once_cell::sync::Lazy;

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

use crate::{copy_timeseries_shallow, EvalConfig, marshal_timeseries_fast, Timeseries, unmarshal_timeseries_fast};
/// import commonly used items from the prelude:
use rand::prelude::*;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::traits::{Timestamp, TimestampTrait};
use crate::utils::MemoryLimiter;

/// The maximum duration since the current time for response data, which is always queried from the
/// original raw data, without using the response cache. Increase this value if you see gaps in responses
/// due to time synchronization issues between this library and data sources. See also
/// -search.disableAutoCacheReset
/// TODO: move to EvalConfig
const cache_timestamp_offset: Duration = Duration::minutes(5);

/// Whether to disable automatic response cache reset if a sample with timestamp outside
/// -search.cachetimestampOffset is inserted into VictoriaMetrics
// disableAutoCacheReset = flag.Bool("search.disableAutoCacheReset", false,

static ROLLUP_RESULT_CACHE_KEY_PREFIX: Lazy<u64> = Lazy::new(|| random::<u64>() );
static NEED_ROLLUP_CACHE_RESET: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new( false ));

const CHECK_ROLLUP_RESULT_CACHE_RESET_INTERVAL: Duration = Duration::seconds(5);

fn get_default_cache_size() -> u64 {
    let mut n = memory_limit().unwrap();
    if n <= 0 {
        n = 1024 * 1024;
    }
    n
}

pub struct RollupResultCache {
    cache: Cache,
    pub memory_limiter: MemoryLimiter,
    max_marshaled_size: u64,
    cache_key_suffix: AtomicU64
}


impl Default for RollupResultCache {
    fn default() -> Self {
        let size = get_default_cache_size();
        Self::new(size as usize)
    }
}


impl RollupResultCache {
    // todo: pass in cache
    pub fn new(size: usize) -> Self {
        let mut rng = thread_rng();
        let suffix: u64 = rng.gen_range((1 << 16)..(1 << 31));
        Self {
            cache: (),
            memory_limiter: MemoryLimiter::new(size as usize),
            max_marshaled_size: size as u64 / 4_u64,
            cache_key_suffix: AtomicU64::new( suffix )
        }
    }

    pub fn reserve_memory(&mut self, size: usize) -> bool {
        self.memory_limiter.get(size)
    }

    pub fn release_memory(&mut self, size: usize) -> RuntimeResult<()> {
        self.memory_limiter.put(size)
    }


    pub fn get(&self, ec: &EvalConfig, expr: &Expression, window: i64) -> RuntimeResult<(Option<Vec<Timeseries>>, i64)> {
        if !ec.may_cache() {
            return Ok((None, ec.start));
        }

        // Obtain tss from the cache.
        let mut bb = get_pooled_buffer(2048);
        let buf = bb.deref_mut();

        marshal_rollup_result_cache_key(bb.deref_mut(),
                                        expr, window, ec.step, &ec.enforced_tag_filterss);
        let metainfo_buf = self.cache.get(None, buf);
        if metainfo_buf.len() == 0 {
            return Ok((None, ec.start));
        }
        let mut mi: RollupResultCacheMetaInfo;

        match RollupResultCacheMetaInfo::from_buf(metainfo_buf) {
            Err(err) => {
                panic!("BUG: cannot unmarshal RollupResultCacheMetainfo; it looks like it was improperly saved");
            },
            Ok(m) => mi = m
        }

        let key = mi.get_best_key(ec.start, ec.end);
        if key.prefix == 0 && key.suffix == 0 {
            return Ok((None, ec.start));
        }
        bb.clear();
        key.marshal(bb.deref_mut());

        let compressed_result_buf = get_pooled_buffer(2048);

        self.cache.get_big(compressed_result_buf, bb.as_slice());
        if compressed_result_buf.len() == 0 {
            mi.remove_key(key);
            mi.marshal(&mut metainfo_buf);
            marshal_rollup_result_cache_key(bb.deref_mut(), expr,
                    window, ec.step, &ec.enforced_tag_filterss);
            self.cache.set(bb, metainfo_buf);
            return Ok((None, ec.start));
        }
        // Decompress into newly allocated byte slice, since tss returned from unmarshalTimeseriesFast
        // refers to the byte slice, so it cannot be returned to the resultBufPool.

        let mut tss: Vec<Timeseries>;

        match decompress_lz4(&compressed_result_buf) {
            Ok(uncompressed) => {
                match unmarshal_timeseries_fast(&uncompressed) {
                    Ok(tss) => tss,
                    Err(err) => {
                        let msg = format!("BUG: cannot unmarshal timeseries from RollupResultCache:; it looks like it was improperly saved");
                        return Err(RuntimeError::from(msg))
                    }
                }
            },
            Err(err) => {
                return Err(RuntimeError::from(
                    format!("BUG: cannot decompress resultBuf from RollupResultCache: {}; it looks like it was improperly saved", err)
                ));
            }
        };

        // Extract values for the matching timestamps
        let mut timestamps = &tss[0].timestamps;
        let mut i = 0;
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

        let mut j = timestamps.len() - 1;
        while j >= 0 && timestamps[j] > ec.end {
            j -= 1;
        };
        j += 1;
        if j <= i {
            // no matches.
            return Ok((None, ec.start));
        }

        timestamps.truncate(j-1);
        timestamps.drain(0..i);
        for ts in tss.iter_mut() {
            ts.values.truncate(j - 1);
            let _ = ts.values.drain(0..i);
        }

        let timestamps = &tss[0].timestamps;
        let new_start = timestamps[timestamps.len() - 1] + ec.step;
        return Ok((Some(tss), new_start));
    }

    pub fn put(&mut self, ec: &EvalConfig, expr: &Expression, window: i64, tss: &Vec<Timeseries>) -> RuntimeResult<()> {
        if tss.len() == 0 || !ec.may_cache() {
            return Ok(());
        }

        // Remove values up to currentTime - step - cache_timestamp_offset,
        // since these values may be added later.
        let timestamps = &tss[0].timestamps;
        let deadline = (Timestamp::now() as f64 / 1e6_f64) as i64 - ec.step - cache_timestamp_offset.num_milliseconds();
        let mut i = timestamps.len() - 1;
        while i >= 0 && timestamps[i] > deadline {
            i -= 1;
        }
        i += 1;
        if i == 0 {
            // Nothing to store in the cache.
            return Ok(());
        }

        let mut start = timestamps[0];
        let mut end = timestamps[timestamps.len() - 1];

        // shadow variable because of the following:
        let rvs = tss;
        if i < timestamps.len() {
            end = timestamps[i];
            // Make a copy of tss and remove unfit values
            let mut _rvs = copy_timeseries_shallow(tss);
            for ts in _rvs.iter_mut() {
                ts.timestamps.resize(i, 0);
                ts.values.resize(i, 0.0);
            }
            tss = rvs
        }

        // Store tss in the cache.
        let mut metainfo_key = get_pooled_buffer(512);
        let mut metainfo_buf = get_pooled_buffer(1024);

        marshal_rollup_result_cache_key(metainfo_key.deref_mut(),
                                        expr,
                                        window,
                                        ec.step,
                                        &ec.enforced_tag_filterss);

        self.cache.get(&mut metainfo_buf, metainfo_key);

        let suffix = self.cache_key_suffix.fetch_add(0, Ordering::Relaxed);

        let mut mi = if metainfo_buf.len() > 0 {
            match RollupResultCacheMetaInfo::from_buf(metainfo_buf.deref_mut()) {
                Err(e) => {
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

        let key = RollupResultCacheKey::new(suffix);
        metainfo_key.clear();
        key.marshal(metainfo_key.deref_mut());

        self.cache.set_big(metainfo_key, compressed_buf);

        let mut buf_ref = metainfo_buf.deref();
        mi.add_key(key, start, end);
        mi.marshal(&mut metainfo_buf);
        self.cache.set(metainfo_key.deref(), metainfo_buf.deref());
        return Ok(())
    }

    fn next_suffix(&mut self) -> u64 {
        self.cache_key_suffix.fetch_add(1, Ordering::SeqCst)
    }
}

// var resultBufPool bytesutil.ByteBufferPool

/// Increment this value every time the format of the cache changes.
const ROLLUP_RESULT_CACHE_VERSION: u8 = 8;

fn marshal_rollup_result_cache_key(dst: &mut Vec<u8>,
                                   expr: &Expression,
                                   window: i64, 
                                   step: i64, 
                                   etfs: &Vec<Vec<LabelFilter>>) {
    let prefix: u64 = *ROLLUP_RESULT_CACHE_KEY_PREFIX.deref();
    dst.push(ROLLUP_RESULT_CACHE_VERSION);
    marshal_fixed_int(dst, prefix);
    marshal_fixed_int(dst, window);
    marshal_fixed_int(dst, step);

    let str_expr = format!("{}", expr);
    dst.extend(str_expr.into_bytes());

    for (i, etf) in etfs.iter().enumerate() {
        for f in etf {
            dst.extend_from_slice(f.as_string().as_bytes())
        }
        if i + 1 < etfs.len() {
            dst.push(b'|');
        }
    }
}

/// merge_timeseries concatenates b with a and returns the result.
///
/// Preconditions:
/// - a mustn't intersect with b.
/// - a timestamps must be smaller than b timestamps.
///
/// Postconditions:
/// - a and b cannot be used after returning from the call.
pub fn merge_timeseries(a: &[Timeseries], b: &mut Vec<Timeseries>, b_start: i64, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
    let shared_timestamps = ec.get_shared_timestamps();
    if b_start == ec.start {
        // Nothing to merge - b covers all the time range.
        // Verify b is correct.
        for tsB in b.iter_mut() {
            tsB.deny_reuse = true;
            tsB.timestamps = shared_timestamps.clone();
            if tsB.values.len() != tsB.timestamps.len() {
                panic!("BUG: unexpected number of values in b; got {}; want {}",
                              tsB.values.len(), tsB.timestamps.len())
            }
        }
        return Ok(b.into());
    }

    let mut m: HashMap<&str, &Timeseries> = HashMap::with_capacity(a.len());
    // tinyvec ??
    let mut bb = get_pooled_buffer(1024);

    for ts in a.iter() {
        let key = ts.metric_name.marshal_to_string(bb.deref_mut())?;
        m.insert(&key, ts);
        bb.clear();
    }

    let mut rvs: Vec<Timeseries> = Vec::with_capacity(a.len());
    for tsB in b.iter_mut() {
        let mut tmp: Timeseries = Timeseries {
            metric_name: tsB.metric_name.into(),
            deny_reuse: true,
            timestamps: shared_timestamps.clone(),
            values: Vec::with_capacity(shared_timestamps.len())
        };
        // do not use MetricName.copy_from for performance reasons.
        // It is safe to make shallow copy, since tsB must no longer used.

        let key = tsB.metric_name.marshal_to_string(bb.deref_mut())?;

        let k = key.as_str();
        match m.get_mut(k) {
            None => {
                let mut t_start = ec.start;
                while t_start < b_start {
                    tmp.values.push(f64::NAN);
                    t_start += ec.step;
                }
            },
            Some(ts_a) => {
                tmp.values.extend(ts_a.values.into_iter());
                m.remove(k);
            }
        }

        tmp.values.append(&mut tsB.values);
        if tmp.values.len() != tmp.timestamps.len() {
            panic!("BUG: unexpected values after merging new values; got {}; want {}",
                   tmp.values.len(), tmp.timestamps.len());
        }
        rvs.push(tmp);
        bb.clear();
    }

    // Copy the remaining timeseries from m.
    for (_, mut tsA) in m.iter_mut() {
        let mut tmp = Timeseries::default();
        tmp.deny_reuse = true;
        tmp.timestamps = shared_timestamps.clone();
        // do not use MetricName.CopyFrom for performance reasons.
        // It is safe to make shallow copy, since tsA must no longer used.
        tmp.metric_name = tsA.metric_name.into();
        tmp.values.extend_from_slice(&mut *tsA.values);

        let mut t_start = b_start;
        while t_start <= ec.end {
            tmp.values.push(f64::NAN);
            t_start += ec.step;
        }
        if tmp.values.len() != tmp.timestamps.len() {
            panic!("BUG: unexpected values in the result after adding cached values; got {}; want {}",
                          tmp.values.len(), tmp.timestamps.len())
        }
        rvs.push(tmp);
    }

    Ok(rvs)
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
        for entry in self.entries {
            entry.marshal(dst);
        }
    }

    fn unmarshal(buf: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfo, &[u8])> {
        let mut src = buf;

        let mut entries_len: usize = 0;
        match unmarshal_fixed_int::<usize>(src) {
            Ok((v, tail)) => {
                entries_len = v;
                src = tail;
            },
            Err(err) => {
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
            panic!("BUG: start cannot exceed end; got {} vs {}", start, end)
        }
        for entry in self.entries.iter() {
            if start >= entry.start && end <= entry.end {
                return true
            }
        }
        return false
    }

    fn get_best_key(&self, start: i64, end: i64) -> RollupResultCacheKey {
        if start > end {
            panic!("BUG: start cannot exceed end; got {} vs {}", start, end)
        }
        let mut best_key: RollupResultCacheKey = RollupResultCacheKey::default();
        let mut d_max: i64 = 0;
        for e in self.entries {
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
        return best_key;
    }

    fn add_key(&mut self, key: RollupResultCacheKey, start: i64, end: i64) {
        if start > end {
            panic!("BUG: start cannot exceed end; got {} vs {}", start, end)
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
    fn new(start: i64, end: i64) -> Self {
        RollupResultCacheMetaInfoEntry {
            start,
            end,
            key: RollupResultCacheKey::default()
        }
    }

    fn read(src: &[u8]) -> RuntimeResult<(RollupResultCacheMetaInfoEntry, &[u8])> {
        let mut entry = Self::default();
        let tail = entry.unmarshal(src)?;
        Ok((entry, tail))
    }

    fn marshal(&self, dst: &mut Vec<u8>) {
        marshal_var_int(dst, self.start);
        marshal_var_int(dst, self.end);
        self.key.marshal(dst);
    }

    fn unmarshal(&mut self, src: &[u8]) -> RuntimeResult<&[u8]> {

        if src.len() < 8 {
            return Err(RuntimeError::SerializationError(
                format!("cannot unmarshal start from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }
        
        let mut src = src;
        
        match unmarshal_var_int::<i64>(src) {
            Err(err) => {
                return Err(RuntimeError::SerializationError(
                    format!("cannot unmarshal start: {:?}", err)
                ));
            },
            Ok((start, tail)) => {
                self.start = start;
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
                self.end = start;
                src = tail;
            }
        }

        self.key.unmarshal(src);

        Ok(src)
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

    fn unmarshal(src: &[u8]) -> RuntimeResult<(RollupResultCacheKey, &[u8])> {
        if src.len() < 8 {
            return Err(RuntimeError::SerializationError(
                format!("cannot unmarshal key prefix from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }
        let mut cursor: &[u8];
        let mut prefix: u64;

        match unmarshal_var_int::<u64>(src) {
            Err(err) => {
                return Err(RuntimeError::SerializationError("cannot unmarshal prefix".to_string()));
            },
            Ok((val, tail)) => {
                prefix = val;
                cursor = tail;
            }
        }

        let mut suffix: u64;

        if src.len() < 8 {
            return Err(RuntimeError::from(
                format!("cannot unmarshal key suffix from {} bytes; need at least {} bytes", src.len(), 8)
            ));
        }

        match unmarshal_var_int::<u64>(cursor) {
            Err(err) => {
                return Err(RuntimeError::SerializationError("error unmarshaling suffix".to_string()));
            },
            Ok((val, tail)) => {
                suffix = val;
                cursor = tail;
            }
        }

       Ok((RollupResultCacheKey{ prefix, suffix }, cursor))
    }
}