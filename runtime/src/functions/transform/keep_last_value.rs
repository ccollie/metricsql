use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeResult};

pub(crate) fn keep_last_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = tfa.get_param_series(0)?;
    for ts in series.iter_mut() {
        if ts.is_empty() {
            continue;
        }
        let mut last_value = ts.values[0];
        for v in ts.values.iter_mut() {
            if !v.is_nan() {
                last_value = *v;
                continue;
            }
            *v = last_value
        }
    }

    Ok(series)
}
