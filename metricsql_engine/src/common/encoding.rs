use integer_encoding::VarInt;

use crate::{RuntimeError, RuntimeResult};

/// appends marshaled v to dst and returns the result.
pub fn marshal_var_int<T: VarInt>(dst: &mut Vec<u8>, v: T) {
    let len = dst.len();
    dst.resize(len + v.required_space(), 0);
    let _ = v.encode_var(&mut dst[len..]);
}

/// unmarshal_var returns unmarshalled int from src.
pub fn unmarshal_var_int<T: VarInt>(src: &[u8]) -> RuntimeResult<(T, &[u8])> {
    match T::decode_var(src) {
        Some((v, ofs)) => Ok((v, &src[ofs..])),
        _ => Err(RuntimeError::SerializationError(
            "Error decoding var int".to_string(),
        )),
    }
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_u64(dst: &mut Vec<u8>, v: u64) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u64 from src and returns the remaining tail from src.
#[inline]
pub fn unmarshal_var_u64(src: &[u8]) -> RuntimeResult<(u64, &[u8])> {
    unmarshal_var_int::<u64>(src)
}

#[inline]
pub fn marshal_var_i64(dst: &mut Vec<u8>, v: i64) {
    marshal_var_int(dst, v)
}

/// unmarshal_var_i64 returns unmarshalled int64 from src and returns
/// the remaining tail from src.
#[inline]
pub fn unmarshal_var_i64(src: &[u8]) -> RuntimeResult<(i64, &[u8])> {
    unmarshal_var_int::<i64>(src)
}

/// appends marshaled v to dst and returns the result.
#[inline]
pub fn marshal_var_usize(dst: &mut Vec<u8>, v: usize) {
    marshal_var_int(dst, v)
}

/// returns unmarshalled u16 from src and returns the remaining tail from src.
pub fn unmarshal_var_usize(src: &[u8]) -> RuntimeResult<(usize, &[u8])> {
    unmarshal_var_int::<usize>(src)
}

fn map_unmarshal_err(e: RuntimeError, what: &str) -> RuntimeError {
    let msg = format!("error reading {}: {:?}", what, e);
    RuntimeError::SerializationError(msg)
}

pub(crate) fn write_usize(slice: &mut Vec<u8>, size: usize) {
    marshal_var_usize(slice, size);
}

pub(crate) fn read_usize<'a>(
    compressed: &'a [u8],
    context: &str,
) -> RuntimeResult<(&'a [u8], usize)> {
    let (size, tail) =
        unmarshal_var_usize(compressed).map_err(|e| map_unmarshal_err(e, context))?;
    Ok((tail, size))
}

pub(crate) fn write_string(buf: &mut Vec<u8>, s: &str) {
    marshal_var_usize(buf, s.len());
    if !s.is_empty() {
        buf.extend_from_slice(s.as_bytes());
    }
}

pub(crate) fn read_string<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], String)> {
    if let Some((len, n)) = u64::decode_var(slice) {
        let len = len as usize;
        let tail = &slice[n..];
        if tail.len() < len {
            return Err(RuntimeError::SerializationError(format!(
                "unexpected end of data when reading {what}: len={len}, tail.len()={}",
                tail.len()
            )));
        }
        let s = String::from_utf8_lossy(&tail[..len]).to_string();
        let tail = &tail[len..];
        Ok((tail, s))
    } else {
        Err(RuntimeError::SerializationError(format!(
            "cannot decode varint from {}",
            what
        )))
    }
}

pub(crate) fn read_u64<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], u64)> {
    let (v, tail) = unmarshal_var_u64(slice).map_err(|e| map_unmarshal_err(e, what))?;
    Ok((tail, v))
}

pub(crate) fn read_i64<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], i64)> {
    let (v, tail) = unmarshal_var_i64(slice).map_err(|e| map_unmarshal_err(e, what))?;
    Ok((tail, v))
}
