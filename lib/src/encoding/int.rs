use crate::error::{Error, Result};
use byte_pool::{Block, BytePool};
use integer_encoding::{FixedInt, VarInt};
use once_cell::sync::Lazy;

/// unmarshal_uint16 returns unmarshaled: u32 from src.
pub fn unmarshal_uint16(src: &[u8]) -> Result<(u16, &[u8])> {
    unmarshal_var_int::<u16>(src)
}

/// unmarshal_uint32 returns unmarshaled: u32 from src.
pub fn unmarshal_uint32(src: &[u8]) -> Result<(u32, &[u8])> {
    unmarshal_var_int::<u32>(src)
}

/// marshal_uint64 appends marshaled v to dst and returns the result.
pub fn marshal_uint64(mut dst: Vec<u8>, u: u64) {
    let mut buf: [u8; 10] = [0; 10];
    let size = u.encode_var(&mut buf);
    dst.extend_from_slice(&buf[0..size]);
}

/// appends marshaled v to dst and returns the result.
pub fn marshal_var_int<T: VarInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf: [u8; 10] = [0; 10];
    let size = v.encode_var(&mut buf);
    dst.extend_from_slice(&buf[0..size]);
}

/// unmarshal_int16 returns unmarshaled int16 from src.
pub fn unmarshal_int16(src: &[u8]) -> Result<(i16, &[u8])> {
    unmarshal_var_int::<i16>(src)
}

/// marshal_int64 appends marshaled v to dst and returns the result.
pub fn marshal_int64(dst: &mut Vec<u8>, v: i64) {
    let mut buf = [0_u8; 10];
    let b = v.encode_var(&mut buf);
    dst.extend_from_slice(&buf[0..b]);
}

/// marshal_fixed_int appends marshaled v to dst and returns the result.
pub fn marshal_fixed_int<T: FixedInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf = [0_u8; 10];
    v.encode_fixed(&mut buf);
    dst.extend_from_slice(&buf[0..T::REQUIRED_SPACE]);
}

/// unmarshal_int64 returns unmarshaled int64 from src.
pub fn unmarshal_fixed_int<T: FixedInt>(src: &[u8]) -> Result<(T, &[u8])> {
    if src.len() < T::REQUIRED_SPACE {
        return Err(Error::new(format!(
            "At least {} bytes required decoding int. Got {}",
            T::REQUIRED_SPACE,
            src.len()
        )));
    }
    Ok((T::decode_fixed(src), &src[T::REQUIRED_SPACE..]))
}

/// unmarshal_var returns unmarshaled int from src.
pub fn unmarshal_var_int<T: VarInt>(src: &[u8]) -> Result<(T, &[u8])> {
    match T::decode_var(src) {
        Some((v, ofs)) => Ok((v, &src[ofs..])),
        _ => Err(Error::new("Error decoding var int")),
    }
}

/// unmarshal_var_int64 returns unmarshaled int64 from src and returns
/// the remaining tail from src.
pub fn unmarshal_var_int64(src: &[u8]) -> Result<(i64, &[u8])> {
    unmarshal_var_int::<i64>(src)
}

const TEMP_BUF_LEN: usize = 512;

/// appends marshaled us to dst.
pub fn marshal_var_int_array<T: VarInt>(dst: &mut Vec<u8>, us: &[T]) {
    let mut buf: [u8; TEMP_BUF_LEN] = [0; TEMP_BUF_LEN];

    let mut ofs: usize = 0;

    for u in us {
        if ofs + 10 >= TEMP_BUF_LEN {
            dst.reserve(ofs);
            dst.extend_from_slice(&buf[0..ofs]);
            ofs = 0;
        }
        ofs += u.encode_var(&mut buf[ofs..]);
    }

    if ofs > 0 {
        dst.extend_from_slice(&buf[0..ofs])
    }
}

/// unmarshals dst.len() values from src to dst
/// and returns the remaining tail from src.
pub fn unmarshal_varint_slice<'a, T: VarInt>(dst: &mut [T], src: &'a [u8]) -> Result<&'a [u8]> {
    let mut ofs: usize = 0;
    let mut i = dst.len();
    let mut j = 0;

    while ofs < src.len() && i > 0 {
        let cursor = &src[ofs..];
        match T::decode_var(cursor) {
            Some((x, size)) => {
                ofs += size;
                dst[j] = x;
            }
            None => {
                let msg = format!("unexpected end of encoded varint at byte {}", ofs);
                return Err(Error::new(msg));
            }
        }
        i -= 1;
        j += 1;
    }
    if i > 0 {
        let msg = format!("unexpected {} items; got {}", dst.len(), dst.len() - i);
        return Err(Error::new(msg));
    }
    Ok(&src[ofs..])
}

/// unmarshal_bytes returns unmarshaled bytes from src.
pub fn unmarshal_bytes(src: &[u8]) -> Result<(&[u8], &[u8])> {
    match unmarshal_var_int::<usize>(src) {
        Ok((n, _)) => {
            if src.len() < n {
                return Err(Error::from(format!(
                    "src is too short for reading string with size {}; src.len()={}",
                    n,
                    src.len()
                )));
            }
            Ok(src.split_at(n))
        }
        Err(err) => Err(Error::new(format!("cannot unmarshal string size: {}", err))),
    }
}

static F64_POOL: Lazy<BytePool<Vec<f64>>> = Lazy::new(BytePool::<Vec<f64>>::new);

static INT64_POOL: Lazy<BytePool<Vec<i64>>> = Lazy::new(BytePool::<Vec<i64>>::new);

/// get_int64s returns an int64 slice with the given size.
pub fn get_int64s(size: usize) -> Block<'static, Vec<i64>> {
    let mut v = INT64_POOL.alloc(size);
    v.clear();
    v
}

/// get_int64s returns an int64 slice with the given size.
pub fn get_float64s(size: usize) -> Block<'static, Vec<f64>> {
    let mut v = F64_POOL.alloc(size);
    v.clear();
    v
}
