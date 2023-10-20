use integer_encoding::{FixedInt, VarInt};

use crate::error::{Error, Result};

/// marshal_fixed_int appends marshaled v to dst and returns the result.
pub fn marshal_fixed_int<T: FixedInt>(dst: &mut Vec<u8>, v: T) {
    let mut buf = [0_u8; 16];
    v.encode_fixed(&mut buf[..T::ENCODED_SIZE]);
    dst.extend_from_slice(&buf[0..T::ENCODED_SIZE]);
}

/// unmarshal_fixed_int returns unmarshalled int64 from src.
pub fn unmarshal_fixed_int<T: FixedInt>(src: &[u8]) -> Result<(T, &[u8])> {
    match T::decode_fixed(src) {
        None => Err(Error::new(format!(
            "At least {} bytes required decoding int. Got {}",
            T::ENCODED_SIZE,
            src.len()
        ))),
        Some(v) => Ok((v, &src[T::ENCODED_SIZE..])),
    }
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
    let buf: [u8; 10] = [0; 10];
    let size = v.encode_var(dst);
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
pub fn marshal_var_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u16 from src and returns the remaining tail from src.
pub fn unmarshal_var_usize(src: &[u8]) -> Result<(usize, &[u8])> {
    unmarshal_var_int::<usize>(src)
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
