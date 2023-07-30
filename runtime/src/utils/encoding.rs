use lib::{
    marshal_var_usize, unmarshal_string_fast, unmarshal_var_i64, unmarshal_var_u64,
    unmarshal_var_usize,
};

use crate::{RuntimeError, RuntimeResult};

fn map_unmarshal_err(e: lib::error::Error, what: &str) -> RuntimeError {
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
    unmarshal_string_fast(slice).map_err(|e| map_unmarshal_err(e, what))
}

pub(crate) fn read_u64<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], u64)> {
    let (v, tail) = unmarshal_var_u64(slice).map_err(|e| map_unmarshal_err(e, what))?;
    Ok((tail, v))
}

pub(crate) fn read_i64<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], i64)> {
    let (v, tail) = unmarshal_var_i64(slice).map_err(|e| map_unmarshal_err(e, what))?;
    Ok((tail, v))
}
