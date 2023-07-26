use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

pub(crate) fn transform_interpolate(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut tss = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in tss.iter_mut() {
        if ts.len() == 0 {
            continue;
        }

        let values = &mut ts.values[0..];

        // skip leading and trailing NaNs
        let mut i = 0;
        while i < values.len() && values[i].is_nan() {
            i += 1;
        }

        let mut j = values.len();
        while j > i && values[j - 1].is_nan() {
            j -= 1;
        }

        let values = &mut ts.values[i..j];

        let mut i = 0;
        let mut prev_value = f64::NAN;
        let mut next_value: f64;

        while i < values.len() {
            let v = values[i];
            if !v.is_nan() {
                i += 1;
                continue;
            }
            if i > 0 {
                prev_value = values[i - 1]
            }
            let mut j = i + 1;
            while j < values.len() && values[j].is_nan() {
                j += 1;
            }
            if j >= values.len() {
                next_value = prev_value
            } else {
                next_value = values[j]
            }
            if prev_value.is_nan() {
                prev_value = next_value
            }
            let delta = (next_value - prev_value) / (j - i + 1) as f64;
            while i < j {
                prev_value += delta;
                values[i] = prev_value;
                i += 1;
            }
        }
    }

    Ok(tss)
}
