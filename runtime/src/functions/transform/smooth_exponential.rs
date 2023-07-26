use crate::functions::arg_parse::{get_float_arg, get_series_arg};
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

pub(crate) fn transform_smooth_exponential(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let sf = get_float_arg(&tfa.args, 1, Some(1.0))?;
    let sf_val = if sf.is_nan() { 1.0 } else { sf.clamp(0.0, 1.0) };

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    for ts in series.iter_mut() {
        let len = ts.values.len();

        // skip NaN and Inf
        let mut i = 0;
        for (j, v) in ts.values.iter().enumerate() {
            if v.is_finite() {
                i = j;
                continue;
            }
            break;
        }

        if i >= len {
            continue;
        }

        let mut avg = ts.values[0];
        i += 1;

        for value in ts.values[i..].iter_mut() {
            if !value.is_nan() {
                avg = avg * (1.0 - sf_val) + *value * sf_val;
                *value = avg;
            }
        }
    }

    Ok(series)
}
