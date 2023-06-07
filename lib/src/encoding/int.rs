
use crate::error::{Error, Result};
use integer_encoding::{FixedInt, VarInt};

/// marshal_fixed_int appends marshaled v to dst and returns the result.
pub fn marshal_fixed_int<T: FixedInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf = [0_u8; 16];
    v.encode_fixed(&mut buf[..T::REQUIRED_SPACE]);
    dst.extend_from_slice(&buf[0..T::REQUIRED_SPACE]);
}

/// unmarshal_int64 returns unmarshalled int64 from src.
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

/// marshal_i16 appends marshaled v to dst and returns the result.
pub fn marshal_i16(dst: &mut Vec<u8>, u: i16) {
    marshal_fixed_int(dst, u);
}

/// unmarshal_i16 returns unmarshalled int16 from src.
pub fn unmarshal_i16(src: &[u8]) -> Result<(i16, &[u8])> {
    unmarshal_fixed_int::<i16>(src)
}

/// marshals a u16 to dst.
pub fn marshal_u16(dst: &mut Vec<u8>, v: u16) {
    marshal_fixed_int::<u16>(dst, v)
}

/// unmarshal_uint16 returns unmarshalled: u16 from src.
pub fn unmarshal_u16(src: &[u8]) -> Result<(u16, &[u8])> {
    unmarshal_fixed_int::<u16>(src)
}

/// marshals a u32 to dst.
pub fn marshal_u32(dst: &mut Vec<u8>, v: u32) {
    marshal_fixed_int::<u32>(dst, v)
}

/// unmarshal_u32 returns unmarshalled: u32 from src.
pub fn unmarshal_u32(src: &[u8]) -> Result<(u32, &[u8])> {
    unmarshal_fixed_int::<u32>(src)
}

/// marshals a usize to dst.
pub fn marshal_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_fixed_int::<usize>(dst, v)
}

/// unmarshal_usize returns unmarshalled usize from src.
pub fn unmarshal_usize(src: &[u8]) -> Result<(usize, &[u8])> {
    unmarshal_fixed_int::<usize>(src)
}

/// marshal_u64 appends marshaled v to dst and returns the result.
pub fn marshal_u64(dst: &mut Vec<u8>, u: u64) {
    marshal_fixed_int::<u64>(dst, u);
}

/// unmarshal_u64 returns unmarshalled u64 from src.
pub fn unmarshal_u64(src: &[u8]) -> Result<(u64, &[u8])> {
    unmarshal_fixed_int::<u64>(src)
}

/// marshal_i64 appends marshaled v to dst and returns the result.
pub fn marshal_i64(dst: &mut Vec<u8>, u: i64) {
    marshal_fixed_int::<i64>(dst, u);
}

/// unmarshal_u64 returns unmarshalled u64 from src.
pub fn unmarshal_i64(src: &[u8]) -> Result<(i64, &[u8])> {
    unmarshal_fixed_int::<i64>(src)
}

/// appends marshaled v to dst and returns the result.
pub fn marshal_var_int<T: VarInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf: [u8; 10] = [0; 10];
    let size = v.encode_var(&mut buf);
    dst.extend_from_slice(&buf[0..size]);
}

/// unmarshal_var returns unmarshalled int from src.
pub fn unmarshal_var_int<T: VarInt>(src: &[u8]) -> Result<(T, &[u8])> {
    match T::decode_var(src) {
        Some((v, ofs)) => Ok((v, &src[ofs..])),
        _ => Err(Error::new("Error decoding var int")),
    }
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_i64(dst: &mut Vec<u8>, v: i64) {
    marshal_var_int(dst, v)
}

/// unmarshal_var_i64 returns unmarshalled int64 from src and returns
/// the remaining tail from src.
pub fn unmarshal_var_i64(src: &[u8]) -> Result<(i64, &[u8])> {
    unmarshal_var_int::<i64>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_u64(dst: &mut Vec<u8>, v: u64) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u64 from src and returns the remaining tail from src.
pub fn unmarshal_var_u64(src: &[u8]) -> Result<(u64, &[u8])> {
    unmarshal_var_int::<u64>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_u32(dst: &mut Vec<u8>, v: u32) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u32 from src and returns the remaining tail from src.
pub fn unmarshal_var_u32(src: &[u8]) -> Result<(u32, &[u8])> {
    unmarshal_var_int::<u32>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_i32(dst: &mut Vec<u8>, v: i32) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled i32 from src and returns the remaining tail from src.
pub fn unmarshal_var_i32(src: &[u8]) -> Result<(i32, &[u8])> {
    unmarshal_var_int::<i32>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_u16(dst: &mut Vec<u8>, v: u16) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u16 from src and returns the remaining tail from src.
pub fn unmarshal_var_u16(src: &[u8]) -> Result<(u16, &[u8])> {
    unmarshal_var_int::<u16>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_i16(dst: &mut Vec<u8>, v: i16) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled i16 from src and returns the remaining tail from src.
pub fn unmarshal_var_i16(src: &[u8]) -> Result<(i16, &[u8])> {
    unmarshal_var_int::<i16>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u16 from src and returns the remaining tail from src.
pub fn unmarshal_var_usize(src: &[u8]) -> Result<(usize, &[u8])> {
    unmarshal_var_int::<usize>(src)
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

/// marshal_bytes appends marshaled b to dst and returns the result.
pub fn marshal_bytes(dst: &mut Vec<u8>, b: &[u8]) {
    let len = b.len();
    marshal_var_int(dst, len);
    if len > 0 {
        dst.extend_from_slice(b);
    }
}

/// unmarshal_bytes returns unmarshalled bytes from src.
/// returns (bytes, remaining tail from src).
pub fn unmarshal_bytes(src: &[u8]) -> Result<(&[u8], &[u8])> {
    match unmarshal_var_int::<usize>(src) {
        Ok((n, tail)) => {
            if src.len() < n {
                return Err(Error::from(format!(
                    "src is too short for reading string with size {}; src.len()={}",
                    n,
                    src.len()
                )));
            }
            let str_bytes = &tail[..n];
            let tail = &tail[n..];
            Ok((str_bytes, tail))
        }
        Err(err) => Err(Error::new(format!("cannot unmarshal string size: {}", err))),
    }
}