use crate::execution::remove_empty_series;
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn transform_drop_empty_series(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() != 1 {
        return Err(RuntimeError::ArgumentError(
            format!("unexpected number of args; got {}; want 1", tfa.args.len()).to_string(),
        ));
    }
    let mut res = get_series_arg(&tfa.args, 0, tfa.ec)?;
    remove_empty_series(&mut res);
    Ok(res)
}
