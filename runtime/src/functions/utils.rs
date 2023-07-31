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

pub fn skip_trailing_nans(values: &[f64]) -> &[f64] {
    let mut i = values.len() - 1;
    while i > 0 && values[i].is_nan() {
        i -= 1;
    }
    &values[0..i + 1]
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
