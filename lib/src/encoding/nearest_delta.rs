use crate::bit_len64;
use crate::encoding::{
    compress::{compress_quantile, decompress_quantile_auto},
    encoding::check_precision_bits,
    int::get_int64s,
};
use crate::error::Error;
use q_compress::CompressorConfig;

/// marshal_int64_nearest_delta encodes src using `nearest delta` encoding
/// with the given precision_bits and appends the encoded value to dst.
///
/// precision_bits must be in the range [1...64], where 1 means 50% precision,
/// while 64 means 100% precision, i.e. lossless encoding.
pub fn marshal_int64_nearest_delta(
    dst: &mut Vec<u8>,
    src: &[i64],
    precision_bits: u8,
) -> Result<i64, Error> {
    if src.is_empty() {
        return Err(Error::from(format!(
            "BUG: src must contain at least 1 item; got {} items",
            src.len()
        )))
    }
    check_precision_bits(precision_bits)?;

    let first_value = src[0];
    let compressed: Vec<u8>;

    if precision_bits == 64 {
        // Fast path.
        let config = CompressorConfig::default()
            .with_use_gcds(false)
            .with_delta_encoding_order(1);

        compressed = compress_quantile(src, config);
    } else {
        // Slower path.
        let mut is = get_int64s(src.len());
        let mut v = first_value;

        let mut trailing_zeros = get_trailing_zeros(v, precision_bits);
        is.push(v);

        for next in src {
            let (d, tzs) = nearest_delta(*next, v, precision_bits, trailing_zeros as u8);
            trailing_zeros = tzs as usize;
            v += d;
            is.push(d);
        }

        let config = CompressorConfig::default().with_use_gcds(false);

        compressed = compress_quantile(&is, config);
    }

    dst.extend_from_slice(&compressed);

    Ok(first_value)
}

/// unmarshal_int64_nearest_delta decodes src using `nearest delta` encoding,
/// appends the result to dst and returns the appended result.
///
/// The first_value must be the value returned from marshal_int64_nearest_delta.
pub(crate) fn unmarshal_int64_nearest_delta(
    dst: &mut Vec<i64>,
    src: &[u8],
    first_value: i64,
    items_count: usize,
) -> Result<(), Error> {
    if items_count < 1 {
        return Err(Error::from(format!(
            "BUG: items_count must be greater than 0; got {}",
            items_count
        )))
    }

    match decompress_quantile_auto::<i64>(src) {
        Err(err) => Err(Error::from(format!(
            "cannot unmarshal nearest delta from {} bytes; src={:?}: {}",
            src.len(),
            src,
            err
        ))),
        Ok(uncompressed) => {
            let mut v = first_value;
            dst.push(v);
            for d in uncompressed {
                v += d;
                dst.push(v);
            }
            Ok(())
        }
    }
}

/// nearest_delta returns the nearest value for (next-prev) with the given
/// precision_bits.
///
/// The second returned value is the number of zeroed trailing bits in the returned delta.
pub(crate) fn nearest_delta(
    next: i64,
    prev: i64,
    precision_bits: u8,
    prev_trailing_zeros: u8,
) -> (i64, u8) {
    let mut d = next - prev;
    if d == 0 {
        // Fast path.
        return (0, dec_if_non_zero(prev_trailing_zeros));
    }

    let mut origin = next;
    if origin < 0 {
        origin = -origin
        // There is no need in handling special case origin = -1<<63.
    }

    let origin_bits = bit_len64(origin as u64) as u8;
    if origin_bits <= precision_bits {
        // Cannot zero trailing bits for the given precision_bits.
        return (d, dec_if_non_zero(prev_trailing_zeros));
    }

    // origin_bits > precision_bits. May zero trailing bits in d.
    let trailing_zeros = origin_bits - precision_bits;
    if trailing_zeros > prev_trailing_zeros + 4 {
        // Probably counter reset. Return d with full precision.
        return (d, prev_trailing_zeros + 2);
    }
    if trailing_zeros + 4 < prev_trailing_zeros {
        // Probably counter reset. Return d with full precision.
        return (d, prev_trailing_zeros - 2);
    }

    // zero trailing bits in d.
    let mut minus = false;
    if d < 0 {
        minus = true;
        d = -d
        // There is no need in handling special case d = -1<<63.
    }
    let mut nd = (d as u64 & ((1 << (64 - 1)) << trailing_zeros)) as i64;
    if minus {
        nd = -nd
    }
    (nd, trailing_zeros)
}

#[inline]
fn dec_if_non_zero(n: u8) -> u8 {
    if n == 0 {
        return 0;
    }
    n - 1
}

pub fn get_trailing_zeros(n: i64, precision_bits: u8) -> usize {
    let mut v = n;
    if v < 0 {
        v = -v
        // There is no need in special case handling for v = -1<<63
    }
    let v_bits = bit_len64(v as u64);
    if v_bits <= precision_bits as usize {
        return 0;
    }
    v_bits - precision_bits as usize
}
