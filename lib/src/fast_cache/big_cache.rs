
/// maxSubvalue_len is the maximum size of subvalue chunk.
///
/// - 16 bytes are for subkey encoding
/// - 4 bytes are for len(key)+len(value) encoding inside fastcache
/// - 1 byte is implementation detail of fastcache
const MAX_SUB_VALUE_LEN: _ = CHUNK_SIZE - 16 - 4 - 1;

/// max_key_len is the maximum size of key.
///
/// - 16 bytes are for (hash + value_len)
/// - 4 bytes are for len(key)+len(subkey)
/// - 1 byte is implementation detail of fastcache
const max_key_len: _ = CHUNK_SIZE - 16 - 4 - 1;

/// set_big sets (k, v) to c where v.len() may exceed 64KB.
///
/// get_big must be used for reading stored values.
///
/// The stored entry may be evicted at any time either due to cache
/// overflow or due to unlikely hash collision.
/// Pass higher maxBytes value to New if the added items disappear
/// frequently.
///
/// It is safe to store entries smaller than 64KB with set_big.
///
/// k and v contents may be modified after returning from set_big.
fn set_big(k: &[u8], v: &[u8]) {
    atomic.AddUint64(&s.big_stats.SetBigCalls, 1);
    if k.len() > max_key_len {
        atomic.AddUint64(&c.bigStats.TooBigKeyErrors, 1);
        return
    }
    let value_len = v.len();
    let value_hash = xxhash.Sum64(v);

    // Split v into chunks with up to 64Kb each.
    // todo: tinyvec
    let mut sub_key = get_pooled_buffer(2048);
    let i: u64 = 0;
    let v = &mut v;
    while v.len() > 0 {
        marshal_var_int(sub_key, value_hash);
        marshal_var_int(sub_key, i);

        i += 1;
        let mut subvalue_len = MAX_SUB_VALUE_LEN;
        if v.len() < subvalue_len {
            subvalue_len = v.len()
        }
        let sub_value = v[0 .. subvalue_len];
        v = v[subvalue_len..];
        self.set(sub_key, sub_value);
        sub_key.clear();
    }

    // Write metavalue, which consists of value_hash and value_len.
    marshal_var_int(sub_key, value_hash);
    marshal_var_int(sub_key, value_len);
    self.set(k, subkey)
}

/// GetBig searches for the value for the given k, appends it to dst
/// and returns the result.
///
/// GetBig returns only values stored via SetBig. It doesn't work
/// with values stored via other methods.
///
/// k contents may be modified after returning from GetBig.
fn get_big(dst: &mut Vec<u8>, k: &[u8]) -> &[u8] {
    atomic.AddUint64(&c.bigStats.GetBigCalls, 1);
    subkey = getSubkeyBuf()
    let dstWasNil = dst == nil;

    // Read and parse metavalue
    subkey.B = c.Get(subkey.B, k);
    if subkey.len() == 0 {
        // Nothing found.
        return dst
    }
    if subkey.len() != 16 {
        atomic.AddUint64(&c.bigStats.InvalidMetavalueErrors, 1)
        return dst
    }
    let value_hash = unmarshalUint64(subkey.B)
    let value_len = unmarshalUint64(subkey.B[8:])

    // Collect result from chunks.
    let dstLen = dst.len();
    if n = dstLen + int(value_len) - cap(dst); n > 0 {
        dst = append(dst[:cap(dst)], make([]byte, n)...)
    }
    let dst = dst[:dstLen]
    let mut i: u64;
    while (dst.len()-dstLen) < value_len {
        subkey.B = marshalUint64(subkey.B, value_hash);
        subkey.B = marshalUint64(subkey.B, uint64(i));
        i += 1;
        let dstNew = self.get(dst, subkey.B);
        if dst_new.len() == dst.len() {
            // Cannot find subvalue
            return &dst[0..dstLen]
        }
        dst = dstNew
    }

    // Verify the obtained value.
    let v = dst[dstLen..];
    if v.len() != value_len {
        atomic.AddUint64(&c.bigStats.Invalidvalue_lenErrors, 1);
        return dst[:dstLen]
    }
    let h = xxhash.Sum64(v);
    if h != value_hash {
        atomic.AddUint64(&c.bigStats.Invalidvalue_hashErrors, 1);
        return dst[:dstLen]
    }
    return dst
}
