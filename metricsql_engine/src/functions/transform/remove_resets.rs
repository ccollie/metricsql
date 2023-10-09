use crate::{RuntimeResult, Timeseries};
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;

pub(crate) fn remove_resets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        remove_counter_resets_maybe_nans(&mut ts.values);
    }
    Ok(series)
}

fn remove_counter_resets_maybe_nans(values: &mut Vec<f64>) {
    let mut i = 0;
    while i < values.len() && values[i].is_nan() {
        i += 1;
    }
    if i > values.len() {
        values.clear();
        return;
    }
    if i > 0 {
        if i == 1 {
            values.remove(i);
        } else {
            values.drain(0..i);
        }
    }
    let mut correction: f64 = 0.0;
    let mut prev_value = values[0];
    for v in values.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let d = *v - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely jitter from `Prometheus HA pairs`.
                // Just substitute v with prev_value.
                *v = prev_value;
            } else {
                correction += prev_value
            }
        }
        prev_value = *v;
        *v += correction;
    }
}
