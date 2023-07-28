use chrono::{TimeZone, Utc};
use chrono_tz::Tz;

use lib::timestamp_ms_to_datetime;
use metricsql::parser::parse_number;

use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, Timeseries};

/// copy_timeseries returns a copy of tss.
pub(super) fn copy_timeseries(tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for src in tss {
        rvs.push(src.clone());
    }
    return rvs;
}

pub(super) fn expect_transform_args_num(
    tfa: &TransformFuncArg,
    expected: usize,
) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    return Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {arg_count}; want {expected}"
    )));
}

pub(super) fn get_num_prefix(s: &str) -> &str {
    let mut s1 = s;
    let mut i = 0;

    if !s.is_empty() {
        let ch = s.chars().next().unwrap();
        if ch == '-' || ch == '+' {
            s1 = &s[1..];
            i += 1;
        }
    }

    let mut has_num = false;
    let mut has_dot = false;

    for ch in s1.chars() {
        if !is_decimal_char(ch) {
            if !has_dot && ch == '.' {
                has_dot = true;
                i += 1;
                continue;
            }
            if !has_num {
                return "";
            }
            return &s[0..i];
        }
        has_num = true;
        i += 1;
    }

    if !has_num {
        return "";
    }
    s
}

fn get_non_num_prefix(s: &str) -> &str {
    for (i, ch) in s.chars().enumerate() {
        if is_decimal_char(ch) {
            return &s[0..i];
        }
    }
    return s;
}

fn must_parse_num(s: &str) -> RuntimeResult<f64> {
    parse_number(s).or_else(|_| Err(RuntimeError::InvalidNumber(s.to_string())))
}

fn is_decimal_char(ch: char) -> bool {
    ch >= '0' && ch <= '9'
}

pub fn get_timezone_offset(zone: &Tz, timestamp_msecs: i64) -> Option<i64> {
    match timestamp_ms_to_datetime(timestamp_msecs) {
        None => None,
        Some(naive) => {
            let in_tz = Utc.from_utc_datetime(&naive).with_timezone(zone);
            Some(in_tz.naive_local().timestamp())
        }
    }
}

#[inline]
/// This exist solely for readability
pub(super) fn clamp_min(val: f64, limit: f64) -> f64 {
    val.min(limit)
}

pub(crate) fn ru(free_value: f64, max_value: f64) -> f64 {
    // ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100
    clamp_min(max_value - clamp_min(free_value, 0_f64), 0_f64) / clamp_min(max_value, 0_f64)
        * 100_f64
}
