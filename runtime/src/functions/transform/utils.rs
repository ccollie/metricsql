use chrono::{TimeZone, Utc};
use chrono_tz::Tz;

use lib::timestamp_ms_to_datetime;
use metricsql::ast::Expr;

use crate::functions::transform::TransformFuncArg;
use crate::{Label, Labels, RuntimeError, RuntimeResult, Timeseries};

/// copy_timeseries returns a copy of tss.
pub(super) fn copy_timeseries(tss: &[Timeseries]) -> Vec<Timeseries> {
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(tss.len());
    for src in tss {
        rvs.push(src.clone());
    }
    rvs
}

pub(super) fn expect_transform_args_num(
    tfa: &TransformFuncArg,
    expected: usize,
) -> RuntimeResult<()> {
    let arg_count = tfa.args.len();
    if arg_count == expected {
        return Ok(());
    }
    Err(RuntimeError::ArgumentError(format!(
        "unexpected number of args; got {arg_count}; want {expected}"
    )))
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

pub fn extract_labels(expr: &Expr) -> Option<Labels> {
    if let Expr::MetricExpression(me) = expr {
        let mut labels = Labels::default();
        for tf in me.label_filters.iter() {
            if tf.label.is_empty() {
                continue;
            }
            if tf.is_regexp() || tf.is_negative() {
                continue;
            }
            labels.push(Label {
                name: tf.label.to_string(),
                value: tf.value.clone(),
            });
        }
        return Some(labels);
    }
    None
}
