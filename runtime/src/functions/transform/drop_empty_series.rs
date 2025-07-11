use crate::execution::remove_empty_series;
use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeError, RuntimeResult};

pub(crate) fn transform_drop_empty_series(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    if tfa.args.len() != 1 {
        return Err(RuntimeError::ArgumentError(
            format!("unexpected number of args; got {}; want 1", tfa.args.len()).to_string(),
        ));
    }
    let mut res = tfa.get_param_series(0)?;
    remove_empty_series(&mut res);
    Ok(res)
}
