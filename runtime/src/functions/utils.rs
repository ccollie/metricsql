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
            i += 1;
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
    let len = values.len();

    if len == 0 {
        return;
    }

    // Slow path: drop nans from values.
    let mut k = 0;
    let mut nan_found = false;
    for i in 0..len {
        let v = values[i];
        if v.is_nan() {
            values[k] = v;
            timestamps[k] = timestamps[i];
            k += 1;
            nan_found = true;
        }
    }

    if nan_found {
        values.truncate(k);
        timestamps.truncate(k);
    }
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

pub(crate) fn float_to_int_bounded(f: f64) -> i64 {
    (f as i64).clamp(i64::MIN, i64::MAX)
}

pub(crate) fn round_to_multiple(n: f64, multiple: f64) -> f64 {
    (n / multiple).round() * multiple
}
