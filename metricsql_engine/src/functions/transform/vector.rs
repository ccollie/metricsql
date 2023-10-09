use crate::{RuntimeResult, Timeseries};
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;

pub(crate) fn vector(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    Ok(series)
}
