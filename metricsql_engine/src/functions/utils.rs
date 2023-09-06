use chrono_tz::Tz;

use metricsql_common::get_local_tz;

use crate::{RuntimeError, RuntimeResult};

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

// todo: move to common metricsql_common
pub(crate) fn parse_timezone(tz_name: &str) -> RuntimeResult<Tz> {
    if tz_name.is_empty() || tz_name.to_ascii_lowercase() == "local" {
        return if let Some(tz) = get_local_tz() {
            Ok(tz)
        } else {
            Err(RuntimeError::ArgumentError(
                "cannot get local timezone".to_string(),
            ))
        };
    }
    match tz_name.parse() {
        Ok(zone) => Ok(zone),
        Err(e) => Err(RuntimeError::ArgumentError(format!(
            "unable to parse tz: {:?}",
            e
        ))),
    }
}