use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeResult};

pub(crate) fn keep_next_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = tfa.get_param_series(0)?;
    for ts in series.iter_mut() {
        if ts.is_empty() {
            continue;
        }
        let mut next_value = *ts.values.last().unwrap();
        for v in ts.values.iter_mut().rev() {
            if !v.is_nan() {
                next_value = *v;
                continue;
            }
            *v = next_value;
        }
    }

    Ok(series)
}
