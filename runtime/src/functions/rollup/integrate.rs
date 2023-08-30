use crate::functions::rollup::{RollupFuncArg, RollupHandler};
use crate::{QueryValue, RuntimeResult};

pub(super) fn new_rollup_integrate(_args: &[QueryValue]) -> RuntimeResult<RollupHandler> {
    Ok(RollupHandler::wrap(rollup_integrate))
}

pub(super) fn rollup_integrate(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    let mut values = &rfa.values[0..];
    let mut timestamps = &rfa.timestamps[0..];
    let mut prev_value = &rfa.prev_value;
    let mut prev_timestamp = rfa.curr_timestamp - rfa.window;
    if prev_value.is_nan() {
        if values.is_empty() {
            return f64::NAN;
        }
        prev_value = &values[0];
        prev_timestamp = timestamps[0];
        values = &values[1..];
        timestamps = &timestamps[1..];
    }

    let mut sum: f64 = 0.0;
    for (v, ts) in values.iter().zip(timestamps.iter()) {
        let dt = (ts - prev_timestamp) as f64 / 1e3_f64;
        sum += prev_value * dt;
        prev_timestamp = *ts;
        prev_value = v;
    }

    let dt = (rfa.curr_timestamp - prev_timestamp) as f64 / 1e3_f64;
    sum += prev_value * dt;
    sum
}
