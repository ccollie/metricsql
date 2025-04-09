use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::types::Timeseries;
use crate::RuntimeResult;

pub(crate) fn vector(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    Ok(series)
}
