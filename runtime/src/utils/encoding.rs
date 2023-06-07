use lib::{marshal_var_usize, unmarshal_bytes, unmarshal_var_usize};
use crate::{RuntimeError, RuntimeResult};

fn map_marshal_err(e: lib::error::Error, what: &str) -> RuntimeError {
    let msg = format!("error writing {}: {:?}", what, e);
    RuntimeError::SerializationError(msg)
}

fn map_unmarshal_err(e: lib::error::Error, what: &str) -> RuntimeError {
    let msg = format!("error reading {}: {:?}", what, e);
    RuntimeError::SerializationError(msg)
}


pub(crate) fn write_usize(slice: &mut Vec<u8>, size: usize) {
    marshal_var_usize(slice, size);
}

pub(crate) fn read_usize<'a>(compressed: &'a [u8], context: &str) -> RuntimeResult<(&'a [u8], usize)> {
    let (size, tail) = unmarshal_var_usize(compressed)
        .map_err(|e| map_unmarshal_err(e, context))?;
    Ok((tail, size))
}

pub(crate) fn write_string(buf: &mut Vec<u8>, s: &str) {
    marshal_var_usize(buf, s.len());
    if s.len() > 0 {
        buf.extend_from_slice(s.as_bytes());
    }
}

pub(crate) fn read_string<'a>(slice: &'a [u8], what: &str) -> RuntimeResult<(&'a [u8], String)> {
    unmarshal_bytes(slice).map_err(|e| map_unmarshal_err(e, what))
        .and_then(|(bytes,slice)| {
            String::from_utf8(bytes.to_vec()).map_err(|e| {
                RuntimeError::SerializationError(
                    format!("invalid utf8 string: {}", e),
                )
            }).map(|s| (slice, s))
        })
}