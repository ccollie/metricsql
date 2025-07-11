use crate::functions::transform::TransformFuncArg;
use crate::types::Timeseries;
use crate::RuntimeResult;

pub(crate) fn smooth_exponential(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let sf = tfa.get_param_scalar(1, "sf", Some(1.0))?;
    let sf_val = if sf.is_nan() { 1.0 } else { sf.clamp(0.0, 1.0) };

    let mut series = tfa.get_param_series(0)?;

    for ts in series.iter_mut() {
        let mut iter = ts.values.iter_mut();
        let mut avg = 0.0;

        // skip NaN and Inf
        for v in iter.by_ref() {
            if v.is_finite() {
                avg = *v;
                break;
            }
        }

        for value in iter {
            if value.is_nan() {
                continue;
            }
            if value.is_infinite() {
                *value = avg;
                continue;
            }
            if !value.is_nan() {
                avg = avg * (1.0 - sf_val) + *value * sf_val;
                *value = avg;
            }
        }
    }

    Ok(series)
}
