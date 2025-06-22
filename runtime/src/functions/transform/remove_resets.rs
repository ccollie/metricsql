use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeResult};

pub(crate) fn remove_resets(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = tfa.get_param_series(0)?;
    for ts in series.iter_mut() {
        remove_counter_resets_maybe_nans(&mut ts.values);
    }
    Ok(series)
}

fn remove_counter_resets_maybe_nans(values: &mut [f64]) {
    let mut start = 0;
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        start = i;
        break;
    }

    let values = &mut values[start..];
    if values.is_empty() {
        return;
    }

    let mut prev_value = values[0];
    let mut correction = 0.0;

    for v in values.iter_mut() {
        if v.is_nan() {
            continue;
        }
        let d = *v - prev_value;
        if d < 0.0 {
            if (-d * 8.0) < prev_value {
                // This is likely a partial counter reset.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2787
                correction += prev_value - *v
            } else {
                correction += prev_value
            }
        }
        prev_value = *v;
        *v += correction;
    }
}
