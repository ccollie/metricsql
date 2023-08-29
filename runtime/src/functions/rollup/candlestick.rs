use crate::functions::rollup::RollupFuncArg;

/// get_candlestick_values returns a subset of rfa.values suitable for rollup_candlestick
///
/// See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/309 for details.
fn get_candlestick_values(rfa: &mut RollupFuncArg) -> &[f64] {
    let curr_timestamp = &rfa.curr_timestamp;
    let timestamps = &rfa.timestamps[0..];
    let mut i = timestamps.len() - 1;

    while i > 0 && timestamps[i] > *curr_timestamp - rfa.window {
        i -= 1;
    }
    if i == 0 {
        return &[];
    }

    &rfa.values[0..i]
}

fn get_first_value_for_candlestick(rfa: &RollupFuncArg) -> f64 {
    if rfa.prev_timestamp + rfa.window >= rfa.curr_timestamp {
        return rfa.prev_value;
    }
    f64::NAN
}

pub(super) fn rollup_open(rfa: &mut RollupFuncArg) -> f64 {
    let v = get_first_value_for_candlestick(rfa);
    if !v.is_nan() {
        return v;
    }
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return f64::NAN;
    }
    values[0]
}

pub(super) fn rollup_close(rfa: &mut RollupFuncArg) -> f64 {
    let values = get_candlestick_values(rfa);
    if values.is_empty() {
        return get_first_value_for_candlestick(rfa);
    }
    values[values.len()]
}

pub(super) fn rollup_high(rfa: &mut RollupFuncArg) -> f64 {
    let mut max = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if max.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        max = values[0];
        start = 1;
    }

    for v in &values[start..] {
        if *v > max {
            max = *v
        }
    }

    max
}

pub(super) fn rollup_low(rfa: &mut RollupFuncArg) -> f64 {
    let mut min = get_first_value_for_candlestick(rfa);
    let values = get_candlestick_values(rfa);
    let mut start = 0;
    if min.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        min = values[0];
        start = 1;
    }
    let vals = &values[start..];
    for v in vals.iter() {
        if v < &min {
            min = *v
        }
    }
    min
}
