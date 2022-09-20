use std::sync::atomic::{AtomicU64, Ordering};
use std::vec::VecDequeue;
use xxhash_rust::xxh3::xxh3_64;

const BUCKETS_COUNT: usize = 512;

const CHUNK_SIZE: usize = 64 * 1024;

const BUCKET_SIZE_BITS: usize = 40;

const GEN_SIZE_BITS: usize = 64 - BUCKET_SIZE_BITS;

const MAX_GEN: u64 = 1 << GEN_SIZE_BITS - 1;

const MAX_BUCKET_SIZE: u64 = 1 << BUCKET_SIZE_BITS;

/// MAX_SUUB_VALUE_LEN is the maximum size of subvalue chunk.
///
/// - 16 bytes are for subkey encoding
/// - 4 bytes are for len(key)+len(value) encoding inside fastcache
/// - 1 byte is implementation detail of fastcache
const MAX_SUB_VALUE_LEN: usize = CHUNK_SIZE - 16 - 4 - 1;

/// MAX_KEY_LEN is the maximum size of key.
///
/// - 16 bytes are for (hash + value_len)
/// - 4 bytes are for len(key)+len(subkey)
/// - 1 byte is implementation detail of fastcache
const MAX_KEY_LEN: usize = CHUNK_SIZE - 16 - 4 - 1;


/// Stats represents cache stats.
///
/// Use Cache.UpdateStats for obtaining fresh stats from the cache.
pub struct Stats {
    /// get_calls is the number of Get calls.
    get_calls: u64,

    /// set_calls is the number of Set calls.
    set_calls: u64,

    /// misses is the number of cache misses.
    misses: u64,

    /// Collisions is the number of cache collisions.
    ///
    /// Usually the number of collisions must be close to zero.
    /// High number of collisions suggest something wrong with cache.
    collisions: u64,

    /// Corruptions is the number of detected corruptions of the cache.
    ///
    /// Corruptions may occur when corrupted cache is loaded from file.
    corruptions: u64,

    /// EntriesCount is the current number of entries in the cache.
    entries_count: u64,

    /// BytesSize is the current size of the cache in bytes.
    bytes_size: u64,

    /// MaxBytesSize is the maximum allowed size of the cache in bytes (aka capacity).
    max_bytes_size: u64,

    /// BigStats contains stats for GetBig/SetBig methods.
    // BigStats
}

impl Stats {
    /// Reset resets s, so it may be re-used again in Cache.UpdateStats.
    pub fn reset(&mut self) {
        self.get_calls = 0;
        self.set_calls = 0;
        self.misses = 0;
        self.collisions = 0;
        self.corruptions = 0;
        self.entries_count = 0;
        self.bytes_size = 0;
        self.max_bytes_size = 0;
    }
}

/// BigStats contains stats for GetBig/SetBig methods.
#[derive(Default, Debug, Clone, Copy)]
pub struct BigStats {
    /// GetBigCalls is the number of GetBig calls.
    get_big_calls: u64,

    /// SetBigCalls is the number of SetBig calls.
    set_big_calls: u64,

    /// TooBigKeyErrors is the number of calls to SetBig with too big key.
    too_big_key_errors: u64,

    /// InvalidMetavalueErrors is the number of calls to GetBig resulting
    /// to invalid metavalue.
    invalid_metavalue_errors: u64,

    /// InvalidValueLenErrors is the number of calls to GetBig resulting
    /// to a chunk with invalid length.
    invalid_value_len_errors: u64,

    /// InvalidValueHashErrors is the number of calls to GetBig resulting
    /// to a chunk with invalid hash value.
    invalid_value_hash_errors: u64,
}

struct CacheInner {
    buckets: [Bucket; bucketsCount],
    // big_stats: BigStats
}

/// Cache is a fast thread-safe in-memory cache optimized for big number
/// of entries.
///
/// It has much lower impact on GC comparing to a simple `HashMap<String,byte[]>`.
///
/// Use New or LoadFromFile* for creating new cache instance.
/// Multiple threads may call any Cache methods on the same cache instance.
///
/// Call reset when the cache is no longer needed. This reclaims the allocated
/// memory.
pub struct Cache {
    inner: CacheInner,
}

/// New returns new cache with the given maxBytes capacity in bytes.
///
/// max_bytes must be smaller than the available RAM size for the app,
/// since the cache holds data in memory.
///
/// If maxBytes is less than 32MB, then the minimum cache capacity is 32MB.
impl Cache {
    pub fn new(max_bytes: usize) -> Self {
        if max_bbytes == 0 {
            panic!("maxBytes must be greater than 0; got {}", max_bytes);
        }
        let max_bucket_bytes = (maxBytes + bucketsCount - 1) / bucketsCount;
        let buckets = [Bucket; BUCKETS_COUNT];
        for i in 0 .. buckets.len() {
            buckets[i].new(max_bucket_bytes)
        }
        return Cache {
            inner: {
                buckets
            }
        }
    }

    /// Set stores (k, v) in the cache.
    ///
    /// Get must be used for reading the stored entry.
    ///
    /// The stored entry may be evicted at any time either due to cache
    /// overflow or due to unlikely hash collision.
    /// Pass higher maxBytes value to `new` if the added items disappear
    /// frequently.
    ///
    /// (k, v) entries with summary size exceeding 64KB aren't stored in the cache.
    /// set_big can be used for storing entries exceeding 64KB.
    ///
    /// k and v contents may be modified after returning from Set.
    pub fn set(&mut self, k: &[u8], v: &[u8]) {
        let h = xxh3_64(k);
        let buckets = self.inner.buckets;
        let idx = h % buckets.len();
        buckets[idx].set(k, v, h)
    }

    /// set_big sets (k, v) to c where v.len() may exceed 64KB.
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
    pub fn set_big(&mut self, k: &[u8], v: &[u8]) {
        self.big_stats.set_big_calls.fetch_add(1, Ordering::Relaxed);
        if k.len() > MAX_KEY_LEN {
            self.big_stats.too_big_key_errors.fetch_add(1, Ordering::Relaxed);
            return
        }
        let value_len = v.len();
        let value_hash = xxh3_64(k);

        // Split v into chunks with up to 64Kb each.
        // todo: tinyvec
        let mut sub_key = get_pooled_buffer(2048);
        let i: u64 = 0;
        let v = &mut v;
        while v.len() > 0 {
            marshal_meta(sub_key, value_hash, i as u64);

            i += 1;
            let mut sub_value_len = MAX_SUB_VALUE_LEN;
            if v.len() < sub_value_len {
                sub_value_len = v.len()
            }
            let sub_value = v[0 ..sub_value_len];
            v = v[sub_value_len..];
            self.set(sub_key, sub_value);
            sub_key.clear();
        }

        // Write metavalue, which consists of value_hash and value_len.
        marshal_meta(sub_key, value_hash, value_len);
        self.set(k, subkey)
    }

    /// get appends value by the key k to dst and returns the result.
    ///
    /// Get allocates new byte slice for the returned value if dst is nil.
    ///
    /// Get returns only values stored in c via Set.
    ///
    /// k contents may be modified after returning from Get.
    pub fn get(&mut self, k: &[u8], dst: &mut Vec<u8>) -> Option<&[u8]> {
        let h = xxh3_64(k);
        let buckets = self.inner.buckets;
        let idx = h % buckets.len();
        let start = dst.len();
        let (dst, found) = buckets[idx].get(dst, k, h, dst);
        if found {
            Some(&dst[start..])
        } else {
            None
        }
    }


    /// Searches for the value for the given k, appends it to dst
    /// and returns the result.
    ///
    /// Returns only values stored via SetBig. It doesn't work
    /// with values stored via other methods.
    ///
    /// k contents may be modified after returning from GetBig.
    pub fn get_big<'a>(&mut self, k: &[u8], dst: &'a mut Vec<u8>) -> &'a [u8] {
        self.big_stats.get_big_calls.fetch_add(1, Ordering::Relaxed);
        let sub_key = getSubkeyBuf();

        // Read and parse metavalue
        sub_key.B = self.get(k, sub_key);
        if sub_key.len() == 0 {
            // Nothing found.
            return dst
        }

        if sub_key.len() != 16 {
            self.big_stats.invalid_metavalue_errors.fetch_add(1, Ordering::Relaxed);
            return dst
        }

        let meta = unmarshal_meta(sub_key);
        if meta.is_none() {
            return  dst
        }

        let (value_hash, value_len) = meta.unwrap();

        // Collect result from chunks.
        let dst_len = dst.len();
        dst.reserve(value_len);

        let dst = &dst[0..dstLen];
        let mut i: u64 = 0;

        while (dst.len() - dst_len) < value_len {
            marshal_meta(sub_key, value_hash, i);
            i += 1;
            let dst_new = self.get(dst, sub_key);
            if dst_new.len() == dst.len() {
                // Cannot find subvalue
                return &dst[0..dstLen]
            }
            dst = dst_new
        }

        // Verify the obtained value.
        let v = &dst[dst_len..];
        if v.len() != value_len {
            atomic.AddUint64(&c.bigStats.invalidvalue_lenErrors, 1);
            return dst[:dstLen]
        }
        let h = xxh3_64(v);
        if h != value_hash {
            atomic.AddUint64(&c.big_stats.invalidvalue_hashErrors, 1);
            return dst[:dstLen]
        }
        return dst
    }
}

#[inline]
fn marshal_meta(buf: &mut Vec<u8>, value_hash: u64, index: u64) {
    marshal_var_int(buf, value_hash);
    marshal_var_int(buf, index);
}

fn unmarshal_meta(src: &[u8]) -> Option<(u64, u64)> {
    if src.len() < 8 {
        return None
    }
    let mut cursor: &[u8];
    let mut value_hash: u64;

    match unmarshal_var_int::<u64>(src) {
        Err(_) => {
            return None;
        },
        Ok((val, tail)) => {
            value_hash = val;
            cursor = tail;
        }
    }

    if src.len() < 8 {
        return None;
    }

    let mut index: u64;

    match unmarshal_var_int::<u64>(cursor) {
        Err(_) => {
            return None
        },
        Ok((val, tail)) => {
            index = val;
            cursor = tail;
        }
    }

    Ok((value_hash, index))
}

struct BucketInner {
    /// chunks is a ring buffer with encoded (k, v) pairs.
    /// It consists of 64KB chunks.
    chunks: Vec<Vec<u8>>,

    /// m maps hash(k) to idx of (k, v) pair in chunks.
    m: HashMap<u64, u64>,

    /// idx points to chunks for writing the next (k, v) pair.
    idx: u64,

    /// gen is the generation of chunks.
    gen: u64,

    get_calls:   AtomicU64,
    set_calls:   AtomicU64,
    misses:      AtomicU64,
    collisions:  AtomicU64,
    corruptions: AtomicU64
}

impl BucketInner {
    pub fn new(bucket_count: usize) -> Self {
        let chunks: Vec<Vec<u8>> = Vec::with_capacity(bucket_count);

        Self {
            chunks,
            m: HashMap::new(),
            idx: 0,
            gen: 0,
            get_calls: AtomicU64::new(0),
            set_calls: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            corruptions: AtomicU64::new(0)
        }
    }

    pub fn reset(&mut self) {
        for chunk in self.chunks.iter_mut() {
            chunk.clear();
        }
        self.idx = 0;
        self.gen = 1;
        self.m.clear();
        self.get_calls.store(0, Ordering::SeqCst);
        self.set_calls.store(0, Ordering::SeqCst);
        self.misses.store(0, Ordering::SeqCst);
        self.collisions.store(0, Ordering::SeqCst);
        self.corruptions.store(0, Ordering::SeqCst);
    }

    fn clean_locked(&mut self) {
        let b_gen = self.gen & ((1 << GEN_SIZE_BITS) - 1);
        let b_idx = self.idx;
        for (k, v) in self.m.iter_mut() {
            let gen = v >> BUCKET_SIZE_BITS;
            let idx = v & ((1 << BUCKET_SIZE_BITS) - 1);
            if (gen+1 == b_gen || gen == MAX_GEN && b_gen == 1) && idx >= b_idx || gen == b_gen && idx < b_idx {
                continue
            }
            self.m.remove(k)
        }
    }

    pub fn set(&mut self, k: &[u8], v: &[u8]) {
        self.set_calls.fetch_add(1, Ordering::Relaxed);
        if k.len() >= (1<<16) || v.len() >= (1<<16) {
            // Too big key or value - its length cannot be encoded
            // with 2 bytes (see below). Skip the entry.
            return
        }
        let mut kv_len_buf: [u8; 4];
        kv_len_buf[0] = (k.len() >> 8) as u8;
        kv_len_buf[1] = k.len() as u8;
        kv_len_buf[2] = (v.len() >> 8) as u8;
        kv_len_buf[3] = v.len() as u8;
        let kv_len = kv_len_buf.len() + k.len() + v.len();
        if kv_len >= CHUNK_SIZE {
            // Do not store too big keys and values, since they do not
            // fit a chunk.
            return
        }

        let mut chunks = self.chunks;
        let mut need_clean = false;
        let mut idx = self.idx;
        let mut idx_new = idx + kv_len;
        let mut chunk_idx = idx / CHUNK_SIZE;
        let chunk_idx_new = idx_new / CHUNK_SIZE;
        if chunk_idx_new > chunk_idx {
            if chunk_idx_new >= self.chunks.len() {
                idx = 0;
                idx_new = kv_len;
                chunk_idx = 0;
                self.gen += 1;
                if self.gen & ((1 << GEN_SIZE_BITS)-1) == 0 {
                    self.gen += 1;
                }
                need_clean = true
            } else {
                idx = chunk_idx_new * CHUNK_SIZE;
                idx_new = idx + kv_len;
                chunk_idx = chunk_idx_new
            }
        }
        if chunk_idx > self.chunks.len() {
            let vec: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);
            self.chunks.resize(chunk_idx, vec)
        }

        let mut chunk = self.chunks[chunk_idx];
        chunk.extend_from_slice(kv_len_buf);
        chunk.extend_from_slice(k);
        chunk.extend_from_slice(v);
        chunks[chunk_idx] = chunk;
        self.m.insert(h, idx | (self.gen << BUCKET_SIZE_BITS));
        self.idx = idx_new;
        if need_clean {
            self.clean_locked()
        }
    }

    pub fn get(&mut self, k: &[u8], h: u64, dst: &mut Vec<u8>) -> Option<&[u8]> {
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let mut found = false;

        if let Some(v) = self.m.get(h) {
            while v > 0 {
                let b_gen = self.gen & ((1 << GEN_SIZE_BITS) - 1);
                let mut gen = v >> BUCKET_SIZE_BITS;
                let mut idx = v & ((1 << BUCKET_SIZE_BITS) - 1);
                if gen == b_gen && idx < self.idx ||
                    gen + 1 == b_gen && idx >= self.idx ||
                    gen == MAX_GEN && b_gen == 1 && idx >= self.idx {
                    let mut chunk_idx = idx / CHUNK_SIZE;
                    if chunk_idx >= self.chunks.len() {
                        // Corrupted data during the load from file. Just skip it.
                        self.corruptions.fetch_add(1, Ordering::Relaxed);
                        break
                    }
                    idx %= CHUNK_SIZE;
                    if idx+4 >= CHUNK_SIZE {
                        // Corrupted data during the load from file. Just skip it.
                        self.corruptions.fetch_add(1, Ordering::Relaxed);
                        break
                    }
                    let mut chunk = self.chunks[chunk_idx];
                    let kv_len_buf = chunk[idx .. idx+4];
                    let key_len = ((kv_len_buf[0] as u64) << 8) | kv_len_buf[1] as u64;
                    let val_len = ((kv_len_buf[2] as u64) << 8) | kv_len_buf[3] as u64;
                    idx += 4;
                    if idx + key_len + val_len >= CHUNK_SIZE {
                        // Corrupted data during the load from file. Just skip it.
                        self.corruptions.fetch_add(1, Ordering::Relaxed);
                        break
                    }

                    let key = &chunk[idx .. idx + key_len];
                    if k == key {
                        idx += key_len;
                        if return_dst {
                            dst.extend_from_slice(&chunk[idx .. idx + val_len]);
                        }
                        found = true
                    } else {
                        self.collisions.fetch_add(1, Ordering::Relaxed)
                    }
                    break
                }
            }
        }

        if !found {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        return dst, found

    }
}

struct Bucket {
    inner: Arc<Mutex<BucketInner>>
}

impl Bucket {
    pub fn new(max_bytes: u64) -> Self {
        if max_bytes == 0 {
            panic!("max_bytes cannot be zero");
        }
        if max_bytes >= MAX_BUCKET_SIZE {
            panic!("too big max_bytes={}; should be smaller than {}", max_bytes, MAX_BUCKET_SIZE);
        }
        let max_chunks = (max_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let chunks: VecDequeue<Vec<u8>> = VecDequeue::with_capacity(max_chunks);
        let data = BucketInner {
            chunks,
            m: HashMap::new(),
            idx: 0,
            gen: 0,
            get_calls: 0,
            set_calls: 0,
            misses: 0,
            collisions: 0,
            corruptions: 0
        };
        let inner = Arc::new(Mutex::new( data ));
        Self {
            inner
        }
    }

    pub fn reset(&mut self) {
        let inner = self.inner.lock().unwrap();
        inner.reset();
    }

    fn clean_locked(&mut self) {
        let inner = self.inner.lock().unwrap();
        inner.clean_locked();
    }

    pub fn set(&mut self, k: &[u8], v: &[u8]) {
        let inner = self.inner.lock().unwrap();
        inner.set(k, v);
    }

    pub fn get(&mut self, k: &[u8], h: u64, return_dst: bool) -> Option<&[u8]> {
        let inner = self.inner.lock().unwrap();
        inner.get(k, h, return_dst)
    }

}