use integer_encoding::VarInt;

use crate::{EncodingError, EncodingResult};

/// marshal_bytes appends marshaled b to dst and returns the result.
pub fn marshal_bytes(b: &[u8], dst: &mut Vec<u8>) {
    let len = b.len() as u64;
    dst.reserve(b.len() + 10);
    len.encode_var(dst);
    if len > 0 {
        dst.extend_from_slice(b);
    }
}

/// unmarshal_bytes returns unmarshalled bytes from src.
/// returns (bytes, remaining tail from src).
pub fn unmarshal_bytes(src: &[u8]) -> EncodingResult<(&[u8], &[u8])> {
    let (len, n) = u64::decode_var(&src[0..]).ok_or_else(|| EncodingError {
        description: "unable to decode timestamp".into(),
    })?;
    let byte_len = src.len() as usize;
    let tail = &src[n..];

    if tail.len() < byte_len {
        let description = format!(
            "src is too short for reading string with size {n}; src.len()={}",
            src.len()
        );
        return Err(EncodingError { description });
    }
    let len = len as usize;
    let str_bytes = &tail[..len];
    let tail = &tail[len..];
    Ok((str_bytes, tail))
}
