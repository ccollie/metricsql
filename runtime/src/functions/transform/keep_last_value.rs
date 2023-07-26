use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

pub(crate) fn transform_keep_last_value(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
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
