use byte_slice_cast::AsByteSlice;

const MAX_N: usize = 8 * 1024;
const INT64_ZEROS: [i64; MAX_N] = [0_i64; MAX_N];
const INT64_ONES: [i64; MAX_N] = [1; MAX_N];
const FLOAT64_ZEROS: [f64; MAX_N] = [0.0; MAX_N];
const FLOAT64_ONES: [f64; MAX_N] = [1.0; MAX_N];

/// append_int64_zeros appends items zeros to dst and returns the result.
///
/// It is faster than the corresponding loop.
pub fn append_int64_zeros(dst: &mut Vec<i64>, items: usize) {
    append_int64_data(dst, items, &INT64_ZEROS)
}

/// AppendInt64Ones appends items ones to dst and returns the result.
///
/// It is faster than the corresponding loop.
pub fn append_int64_ones(dst: &mut Vec<i64>, items: usize) {
    append_int64_data(dst, items, &INT64_ONES);
}

/// Appends items zeros to dst and returns the result.
///
/// It is faster than the corresponding loop.
pub fn append_float64_zeros(dst: &mut Vec<f64>, items: usize) {
    append_float_data(dst, items, &FLOAT64_ZEROS);
}

/// Appends items ones to dst and returns the result.
///
/// It is faster than the corresponding loop.
pub fn append_float64_ones(dst: &mut Vec<f64>, items: usize) {
    append_float_data(dst, items, &FLOAT64_ONES);
}

/// is_int64_zeros checks whether a contains only zeros.
pub fn is_int64_zeros(a: &[i64]) -> bool {
    is_int64_data(a, &INT64_ZEROS)
}

/// is_int64_ones checks whether a contains only ones.
pub(crate) fn is_int64_ones(a: &[i64]) -> bool {
    is_int64_data(a, &INT64_ONES)
}

/// checks whether a contains only zeros.
pub fn is_float64_zeros(a: &[f64]) -> bool {
    is_float64_data(a, &FLOAT64_ZEROS)
}

/// is_float64_ones checks whether a contains only ones.
pub fn is_float64_ones(a: &[f64]) -> bool {
    is_float64_data(a, &FLOAT64_ONES)
}

pub fn append_int64_data(dst: &mut Vec<i64>, count: usize, src: &[i64]) {
    let mut item_count = count;
    while item_count > 0 {
        let mut n = src.len();
        if n > item_count {
            n = item_count
        }
        dst.extend_from_slice(&src[0..n]);
        item_count -= n
    }
}

pub fn append_float_data(dst: &mut Vec<f64>, count: usize, src: &[f64]) {
    let mut items = count;
    while items > 0 {
        let mut n = src.len();
        if n > items {
            n = items
        }
        dst.extend_from_slice(&src[0..n]);
        items -= n
    }
}

pub fn is_int64_data(a: &[i64], data: &[i64]) -> bool {
    if a.is_empty() {
        return true;
    }
    if data.len() != 8 * 1024 {
        panic!("data.len() -> must equal to 8*1024")
    }
    let b = data.as_byte_slice();
    let mut cursor = a;
    while cursor.len() > 0 {
        let mut n = data.len();
        if n > cursor.len() {
            n = cursor.len()
        }
        let x = &cursor[0..n];
        cursor = &cursor[n..];
        let xb = x.as_byte_slice();
        if !xb.eq(&b[0..=xb.len() - 1]) {
            return false;
        }
    }
    true
}

pub fn is_float64_data(a: &[f64], data: &[f64]) -> bool {
    if a.is_empty() {
        return true;
    }
    if data.len() != MAX_N {
        panic!("data.len() -> must equal to 8*1024");
    }
    let b = data.as_byte_slice();
    let mut cursor = a;
    while cursor.len() > 0 {
        let mut n = data.len();
        if n > cursor.len() {
            n = cursor.len()
        }
        let x = &cursor[0..n];
        cursor = &cursor[n..];
        let xb = x.as_byte_slice();
        if !xb.eq(&b[0..=xb.len() - 1]) {
            return false;
        }
    }
    true
}
