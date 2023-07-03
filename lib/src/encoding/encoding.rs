use std::fmt;
use std::iter::repeat;
use std::ops::Deref;

use crate::{fastnum, get_pooled_buffer};
use crate::encoding::compress::{compress_lz4, decompress_lz4};
use crate::encoding::int::{marshal_var_int, unmarshal_var_int};
use crate::encoding::nearest_delta::{marshal_int64_nearest_delta, unmarshal_int64_nearest_delta};
use crate::encoding::nearest_delta2::{
    marshal_int64_nearest_delta2, unmarshal_int64_nearest_delta2,
};
use crate::error::{Error, Result};

/// MIN_COMPRESSIBLE_BLOCK_SIZE is the minimum block size in bytes for trying compression.
///
/// There is no sense in compressing smaller blocks.
const MIN_COMPRESSIBLE_BLOCK_SIZE: usize = 128;

/// MarshalType is the type used for the marshaling.
#[derive(Debug, Clone, PartialOrd, PartialEq, Copy)]
#[non_exhaustive]
pub enum MarshalType {
    /// DeltaConst is used for marshaling constantly changed
    /// time series with constant delta.
    DeltaConst = 1,

    /// Const is used for marshaling time series containing only a single constant.
    Const = 2,

    /// NearestDelta is used instead of Lz4NearestDelta
    /// if compression doesn't help.
    NearestDelta = 3,

    /// NearestDelta2 is used instead of Lz4DNearestDelta2
    /// if compression doesn't help.
    NearestDelta2 = 4,

    /// Lz4DNearestDelta is used for marshaling gauge timeseries.
    Lz4NearestDelta = 5,

    /// Counter timeseries compressed with lz4.
    Lz4NearestDelta2 = 6,
}

impl fmt::Display for MarshalType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use MarshalType::*;
        match self {
            Const => write!(f, "Const")?,
            DeltaConst => write!(f, "DeltaConst")?,
            NearestDelta => write!(f, "NearestDelta")?,
            NearestDelta2 => write!(f, "NearestDelta2")?,
            Lz4NearestDelta => write!(f, "Lz4NearestDelta")?,
            Lz4NearestDelta2 => write!(f, "Lz4NearestDelta2")?,
        }
        Ok(())
    }
}

impl TryFrom<u8> for MarshalType {
    type Error = Error;

    fn try_from(v: u8) -> std::result::Result<Self, Error> {
        match v {
            x if x == MarshalType::Const as u8 => Ok(MarshalType::Const),
            x if x == MarshalType::DeltaConst as u8 => Ok(MarshalType::DeltaConst),
            x if x == MarshalType::NearestDelta as u8 => Ok(MarshalType::NearestDelta),
            x if x == MarshalType::NearestDelta2 as u8 => Ok(MarshalType::NearestDelta2),
            x if x == MarshalType::Lz4NearestDelta as u8 => Ok(MarshalType::Lz4NearestDelta),
            x if x == MarshalType::Lz4NearestDelta2 as u8 => Ok(MarshalType::Lz4NearestDelta2),
            _ => Err(Error::from("Invalid marshal type value: {v}")),
        }
    }
}

/// check_precision_bits makes sure precision_bits is in the range [1..64].
pub(crate) fn check_precision_bits(precision_bits: u8) -> std::result::Result<(), Error> {
    if !(1..=64).contains(&precision_bits) {
        return Err(Error::from(format!(
            "precision_bits must be in the range [1...64]; got {precision_bits}"
        )));
    }
    Ok(())
}

/// marshal_timestamps marshals timestamps, appends the marshaled result
/// to dst and returns the dst.
///
/// timestamps must contain non-decreasing values.
///
/// precision_bits must be in the range [1...64], where 1 means 50% precision,
/// while 64 means 100% precision, i.e. lossless encoding.
pub fn marshal_timestamps(
    dst: &mut Vec<u8>,
    timestamps: &[i64],
    precision_bits: u8,
) -> Result<(MarshalType, i64)> {
    marshal_int64_array(dst, timestamps, precision_bits)
}

/// unmarshal_timestamps unmarshals timestamps from src, appends them to dst
/// and returns the resulting dst.
///
/// first_timestamp must be the timestamp returned from marshal_timestamps.
pub fn unmarshal_timestamps(
    dst: &mut Vec<i64>,
    src: &[u8],
    mt: MarshalType,
    first_value: i64,
    items_count: usize,
) -> Result<()> {
    match unmarshal_int64_array(dst, src, mt, first_value, items_count) {
        Ok(..) => Ok(()),
        Err(err) => {
            let msg = format!(
                "cannot unmarshal timestamps from src.len()={} bytes: {}",
                src.len(),
                err
            );
            Err(Error::from(msg))
        }
    }
}

/// marshal_values marshals values, appends the marshaled result to dst
///
/// precision_bits must be in the range [1...64], where 1 means 50% precision,
/// while 64 means 100% precision, i.e. lossless encoding.
pub fn marshal_values(
    dst: &mut Vec<u8>,
    values: &[i64],
    precision_bits: u8,
) -> Result<(MarshalType, i64)> {
    marshal_int64_array(dst, values, precision_bits)
}

/// unmarshal_values unmarshals values from src, appends them to dst and returns
/// the resulting dst.
///
/// first_value must be the value returned from marshal_values.
pub fn unmarshal_values(
    dst: &mut Vec<i64>,
    src: &[u8],
    mt: MarshalType,
    first_value: i64,
    items_count: usize,
) -> Result<()> {
    match unmarshal_int64_array(dst, src, mt, first_value, items_count) {
        Err(err) => Err(Error::from(format!(
            "cannot unmarshal {items_count} values from src.len()={} bytes: {}",
            src.len(),
            err
        ))),
        _ => Ok(()),
    }
}

pub fn marshal_int64_array(
    dst: &mut Vec<u8>,
    a: &[i64],
    precision_bits: u8,
) -> Result<(MarshalType, i64)> {
    use MarshalType::*;

    if a.is_empty() {
        return Err(Error::from("BUG: a must contain at least one item"));
    }

    if is_const(a) {
        return Ok((Const, a[0]));
    }

    if is_delta_const(a) {
        let first_value = a[0];
        marshal_var_int::<i64>(dst, a[1] - first_value);
        return Ok((DeltaConst, first_value));
    }

    let bb = get_pooled_buffer(2048);
    let first_value: i64;
    let mut mt: MarshalType;

    if is_gauge(a) {
        // Gauge values are better compressed with delta encoding.
        mt = Lz4NearestDelta;
        let mut pb = precision_bits;
        if pb < 6 {
            // Increase precision bits for gauges, since they suffer more
            // from low precision bits comparing to counters.
            pb += 2;
        }
        first_value = marshal_int64_nearest_delta(dst, a, pb)?;
    } else {
        // Non-gauge values, i.e. counters are better compressed with delta2 encoding.
        mt = Lz4NearestDelta2;
        first_value = marshal_int64_nearest_delta2(dst, a, precision_bits)?;
    }

    let orig_len: usize = dst.len();

    // Try compressing the result.
    if bb.len() >= MIN_COMPRESSIBLE_BLOCK_SIZE {
        // quantile compress ???
        let mut compressed = compress_lz4(bb.deref());
        dst.append(&mut compressed);
    }
    if bb.len() < MIN_COMPRESSIBLE_BLOCK_SIZE
        || (dst.len() - orig_len) > (0.9 * bb.len() as f64) as usize
    {
        // Ineffective compression. Store plain data.
        mt = match mt {
            Lz4NearestDelta2 => NearestDelta2,
            Lz4NearestDelta => NearestDelta,
            _ => return Err(Error::from(format!("BUG: unexpected mt={mt}"))),
        };
        dst.extend(bb.as_slice());
    };

    Ok((mt, first_value))
}

pub fn unmarshal_int64_array(
    dst: &mut Vec<i64>,
    src: &[u8],
    mt: MarshalType,
    first_value: i64,
    items_count: usize,
) -> Result<()> {
    use MarshalType::*;

    // Extend dst capacity in order to eliminate memory allocations below.
    dst.reserve(items_count);

    //let mut src = src;

    match mt {
        Lz4NearestDelta => match decompress_lz4(src) {
            Err(err) => Err(Error::from(format!("cannot decompress lz4 data: {}", err))),
            Ok(uncompressed) => {
                match unmarshal_int64_nearest_delta(dst, &uncompressed, first_value, items_count) {
                    Err(err) => {
                        let msg = format!(
                            "cannot unmarshal nearest delta data after decompression: {}",
                            err
                        );
                        Err(Error::from(msg))
                    }
                    Ok(_) => Ok(()),
                }
            }
        },
        Lz4NearestDelta2 => match decompress_lz4(src) {
            Err(err) => Err(Error::from(format!(
                "cannot decompress lz4 data: {:?}",
                err
            ))),
            Ok(uncompressed) => {
                match unmarshal_int64_nearest_delta2(dst, &uncompressed, first_value, items_count) {
                    Err(err) => {
                        let msg = format!(
                            "cannot unmarshal nearest delta2 data after decompression: {}",
                            err
                        );
                        Err(Error::from(msg))
                    }
                    Ok(_) => Ok(()),
                }
            }
        },
        NearestDelta => match unmarshal_int64_nearest_delta(dst, src, first_value, items_count) {
            Err(e) => Err(Error::from(format!(
                "cannot unmarshal nearest delta data: {}",
                e
            ))),
            Ok(_) => Ok(()),
        },
        NearestDelta2 => match unmarshal_int64_nearest_delta2(dst, src, first_value, items_count) {
            Err(e) => Err(Error::from(format!(
                "cannot unmarshal nearest delta2 data: {}",
                e
            ))),
            Ok(_) => Ok(()),
        },
        Const => {
            if !src.is_empty() {
                return Err(Error::from(format!(
                    "unexpected data left in const encoding: {} bytes",
                    src.len()
                )));
            }
            if first_value == 0 {
                fastnum::append_int64_zeros(dst, items_count);
                return Ok(());
            }
            if first_value == 1 {
                fastnum::append_int64_ones(dst, items_count);
                return Ok(());
            }
            dst.reserve(items_count);
            dst.extend(repeat(first_value).take(items_count));
            Ok(())
        }
        DeltaConst => {
            let mut d: i64 = 0;
            match unmarshal_var_int::<i64>(src) {
                Ok((delta, tail)) => {
                    if !tail.is_empty() {
                        return Err(Error::from(format!(
                            "unexpected trailing data after delta const (d={d}): {} bytes",
                            tail.len()
                        )));
                    }
                    d = delta;
                }
                Err(err) => {
                    return Err(Error::new(format!(
                        "cannot unmarshal delta value for delta const: {}",
                        err
                    )))
                }
            };

            let mut v = first_value;
            let mut count = items_count;
            dst.reserve(count);

            while count > 0 {
                dst.push(v);
                count -= 1;
                v += d
            }
            Ok(())
        }
    }
}

/// ensure_non_decreasing_sequence makes sure the first item in a is v_min, the last
/// item in a is v_max and all the items in a are non-decreasing.
///
/// If this isn't the case the a is fixed accordingly.
pub fn ensure_non_decreasing_sequence(a: &mut [i64], v_min: i64, v_max: i64) {
    if v_max < v_min {
        panic!("BUG: v_max cannot be smaller than v_min; got {v_max} vs {v_min}")
    }
    if a.is_empty() {
        return;
    }
    let max = v_max;
    let min = v_min;
    if a[0] != v_min {
        a[0] = min;
    }

    let mut v_prev = a[0];
    for value in a.iter_mut().skip(1) {
        if *value < v_prev {
            *value = v_prev;
        }
        v_prev = *value;
    }

    let mut i = a.len() - 1;
    if a[i] != max {
        a[i] = max;
        if i > 0 {
            i -= 1;
            while i > 0 && a[i] > max {
                a[i] = max;
                i -= 1;
            }
        }
    }
}

/// is_const returns true if a contains only equal values.
pub(crate) fn is_const(a: &[i64]) -> bool {
    if a.is_empty() {
        return false;
    }
    if fastnum::is_int64_zeros(a) {
        // Fast path for array containing only zeros.
        return true;
    }
    if fastnum::is_int64_ones(a) {
        // Fast path for array containing only ones.
        return true;
    }
    let v1 = a[0];
    a.iter().all(|x| *x == v1)
}

/// is_delta_const returns true if a contains counter with constant delta.
pub fn is_delta_const(a: &[i64]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let d1 = a[1] - a[0];
    let mut prev = a[1];
    for next in &a[2..] {
        if *next - prev != d1 {
            return false;
        }
        prev = *next;
    }
    true
}

/// is_gauge returns true if a contains gauge values,
/// i.e. arbitrary changing values.
///
/// It is OK if a few gauges aren't detected (i.e. detected as counters),
/// since misdetected counters as gauges leads to worse compression ratio.
pub fn is_gauge(a: &[i64]) -> bool {
    // Check all the items in a, since a part of items may lead
    // to incorrect gauge detection.

    if a.len() < 2 {
        return false;
    }

    let mut resets = 0;
    let mut v_prev = a[0];
    if v_prev < 0 {
        // Counter values cannot be negative.
        return true;
    }
    for v in &a[1..] {
        if *v < v_prev {
            if *v < 0 {
                // Counter values cannot be negative.
                return true;
            }
            if *v > (v_prev >> 3) {
                // Decreasing sequence detected.
                // This is a gauge.
                return true;
            }
            // Possible counter reset.
            resets += 1;
        }
        v_prev = *v;
    }
    if resets <= 2 {
        // Counter with a few resets.
        return false;
    }

    // Let it be a gauge if resets exceeds a.len()/8,
    // otherwise assume counter.
    resets > (a.len() >> 3)
}

pub fn marshal_string_fast(dst: &mut Vec<u8>, src: &str) {
    let len = src.len();
    dst.reserve(len);
    marshal_var_int(dst, len);
    if len > 0 {
        dst.extend_from_slice(src.as_bytes());
    }
}

pub fn unmarshal_bytes_fast(src: &[u8]) -> Result<(&[u8], &[u8])> {
    if src.is_empty() {
        return Err(Error::from(
            "cannot decode size from src=; it must be at least 2 bytes",
        ));
    }
    return match unmarshal_var_int::<usize>(src) {
        Ok((size, tail)) => {
            if size > 0 && tail.len() < size {
                return Err(Error::from(format!(
                    "too short src; it must be at least {size} bytes")));
            }
            // todo: cap size to prevent issues ????
            let bytes = &tail[0..size];
            let tail = &tail[size..];
            return Ok((tail, bytes));
        }
        Err(err) => Err(Error::from(format!(
            "error unmarshalling string len: {} ",
            err
        ))),
    };
}


pub fn unmarshal_string_fast(src: &[u8]) -> Result<(&[u8], String)> {
    unmarshal_bytes_fast(src)
        .and_then(|(tail, bytes)| {
            let str = std::str::from_utf8(bytes)
                .map_err(|x| Error::from(format!("cannot decode string from bytes: {}", x)))?;
            Ok((tail, str.to_string()))
        })
}