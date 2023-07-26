use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

pub(crate) fn transform_keep_next_value(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
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
