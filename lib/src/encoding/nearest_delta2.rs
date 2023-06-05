use crate::encoding::encoding::check_precision_bits;
use crate::encoding::int::{marshal_var_int, marshal_var_int_array};
use crate::encoding::nearest_delta::{get_trailing_zeros, nearest_delta};
use crate::error::Error;
use crate::{get_int64s, unmarshal_var_int};

/// marshal_int64_nearest_delta2 encodes src using `nearest delta2` encoding
/// with the given precision_bits and appends the encoded value to dst.
///
/// precision_bits must be in the range [1...64], where 1 means 50% precision,
/// while 64 means 100% precision, i.e. lossless encoding.
///
/// Return the first value
pub fn marshal_int64_nearest_delta2(
    dst: &mut Vec<u8>,
    src: &[i64],
    precision_bits: u8,
) -> Result<i64, Error> {
    if src.len() < 2 {
        return Err(Error::from(format!(
            "BUG: src must contain at least 2 items; got {} items",
            src.len()
        )));
    }
    check_precision_bits(precision_bits)?;

    let first_value = src[0];
    let mut d1 = src[1] - src[0];
    marshal_var_int(dst, d1);
    let mut v = src[1];
    let src = &src[2..];

    let block_len = if src.len() < 64 { 64 } else { src.len() };

    let mut is = get_int64s(block_len);
    if precision_bits == 64 {
        // Fast path.
        for next in src {
            let d2 = next - v - d1;
            d1 += d2;
            v += d1;
            is.push(d2);
        }
    } else {
        // Slower path.
        let mut trailing_zeros = get_trailing_zeros(v, precision_bits);
        for next in src {
            let (d2, tzs) = nearest_delta(next - v, d1, precision_bits, trailing_zeros as u8);
            trailing_zeros = tzs as usize;
            d1 += d2;
            v += d1;
            is.push(d2);
        }
    }
    marshal_var_int_array(dst, &is);

    Ok(first_value)
}

/// unmarshal_int64nearest_delta2 decodes src using `nearest delta2` encoding,
/// appends the result to dst and returns the appended result.
///
/// first_value must be the value returned from marshal_int64nearest_delta2.
pub fn unmarshal_int64_nearest_delta2(
    dst: &mut Vec<i64>,
    src: &[u8],
    first_value: i64,
    items_count: usize,
) -> Result<(), Error> {
    if items_count < 2 {
        return Err(Error::from(format!(
            "BUG: items_count must be greater than 1; got {}",
            src.len()
        )));
    }

    let is = get_int64s(items_count - 1);

    return match unmarshal_var_int::<i64>(src) {
        Err(err) => Err(Error::from(format!(
            "cannot unmarshal nearest delta from {} bytes; {}",
            src.len(),
            err
        ))),
        Ok((_, tail)) => {
            if !tail.is_empty() {
                return Err(
                    Error::from(
                        format!("unexpected tail left after unmarshalling {} items from {} bytes; tail size={}; src={:?}; tail={:?}",
                        items_count, src.len(), tail.len(), src, tail)));
            }
            let mut v = first_value;
            let mut d1 = is[0];
            dst.push(v);
            v += d1;
            dst.push(v);
            for i in 1..is.len() {
                let d2 = is[i];
                d1 += d2;
                v += d1;
                dst.push(v)
            }
            Ok(())
        }
    };
}
