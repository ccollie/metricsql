use std::cmp::Ordering;
use std::ops::DerefMut;

use tinyvec::TinyVec;

use lib::get_float64s;

/// mode_no_nans returns mode for a.
///
/// It is expected that a doesn't contain NaNs.
///
/// The function modifies contents for a, so the caller must prepare it accordingly.
///
/// See https://en.wikipedia.org/wiki/Mode_(statistics)
pub fn mode_no_nans(prev_value: f64, a: &mut Vec<f64>) -> f64 {
    let mut prev_value = prev_value;
    if a.len() == 0 {
        return prev_value;
    }
    a.sort_by(|a, b| a.total_cmp(b));
    let mut j: i32 = -1;
    let mut i: i32 = 0;

    let mut d_max = 0;
    let mut mode = prev_value;
    for v in a.iter_mut() {
        if prev_value == *v {
            continue;
        }
        let d = i - j;
        if d > d_max || mode.is_nan() {
            d_max = d;
            mode = prev_value;
        }
        j = i;
        i += 1;
        prev_value = *v;
    }
    let d = a.len() as i32 - j;
    if d > d_max || mode.is_nan() {
        mode = prev_value
    }
    return mode;
}

pub fn remove_nan_values_in_place(values: &mut Vec<f64>, timestamps: &mut Vec<i64>) {
    if !values.iter().any(|x| x.is_nan()) {
        return;
    }

    // Slow path: drop nans from values.
    let mut k = 0;
    for i in 0..values.len() {
        let v = values[i];
        if v.is_nan() {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
        }
    }

    values.truncate(k);
    timestamps.truncate(k);
}

#[inline]
pub fn get_first_non_nan_index(values: &[f64]) -> usize {
    for (index, v) in values.iter().enumerate() {
        if !v.is_nan() {
            return index;
        }
    }
    0
}

pub fn skip_leading_nans(values: &[f64]) -> &[f64] {
    let i = get_first_non_nan_index(values);
    return &values[i..];
}

pub fn skip_trailing_nans(values: &[f64]) -> &[f64] {
    let mut i = values.len() - 1;
    while i > 0 && values[i].is_nan() {
        i -= 1;
    }
    return &values[0..i + 1];
}

#[inline]
pub fn get_last_non_nan_index(values: &[f64]) -> usize {
    let mut i = values.len() - 1;
    while i > 0 && values[i].is_nan() {
        i -= 1;
    }
    i
}

/// quantiles calculates the given phis from originValues without modifying originValues, appends
/// them to qs and returns the result.
pub fn quantiles(qs: &mut [f64], phis: &[f64], origin_values: &[f64]) {
    if origin_values.len() <= 64 {
        let mut vec = tiny_vec!([f64; 64]);
        prepare_tv_for_quantile_float64(&mut vec, origin_values);
        return quantiles_sorted(qs, phis, &vec);
    }

    let mut block = get_float64s(phis.len());
    let a = block.deref_mut();
    prepare_for_quantile_float64(a, origin_values);
    quantiles_sorted(qs, phis, a)
}

/// calculates the given phi from origin_values without modifying origin_values
pub fn quantile(phi: f64, origin_values: &[f64]) -> f64 {
    // todo: smallvec
    let mut block = get_float64s(origin_values.len());
    prepare_for_quantile_float64(&mut block, origin_values);
    quantile_sorted(phi, &block)
}

/// prepare_for_quantile_float64 copies items from src to dst but removes NaNs and sorts the dst
fn prepare_for_quantile_float64(dst: &mut Vec<f64>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue;
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// copies items from src to dst but removes NaNs and sorts the dst
fn prepare_tv_for_quantile_float64(dst: &mut TinyVec<[f64; 64]>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue;
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// calculates the given phis over a sorted list of values, appends them to qs and returns the result.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub fn quantiles_sorted(qs: &mut [f64], phis: &[f64], values: &[f64]) {
    for (i, phi) in phis.iter().enumerate() {
        qs[i] = quantile_sorted(*phi, values);
    }
}

/// quantile_sorted calculates the given quantile over a sorted list of values.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
    if values.len() == 0 || phi.is_nan() {
        return f64::NAN;
    }
    if phi < 0.0 {
        return f64::NEG_INFINITY;
    }
    if phi > 1.0 {
        return f64::INFINITY;
    }
    let n = values.len();
    let rank = phi * (n - 1) as f64;

    let lower_index = std::cmp::max(0, rank.floor() as usize);
    let upper_index = std::cmp::min(n - 1, lower_index + 1) as usize;

    let weight = rank - rank.floor();
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
}

pub(crate) fn float_to_int_bounded(f: f64) -> i64 {
    (f as i64).clamp(i64::MIN, i64::MAX)
}
