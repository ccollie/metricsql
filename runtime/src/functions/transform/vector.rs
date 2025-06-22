use crate::functions::transform::TransformFuncArg;
use crate::types::Timeseries;
use crate::RuntimeResult;

pub(crate) fn vector(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    tfa.get_param_series(0)
}
