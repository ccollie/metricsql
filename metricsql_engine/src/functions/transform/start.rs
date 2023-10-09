use crate::{RuntimeResult, Timeseries};
use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;

#[inline]
pub(crate) fn transform_start(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.start as f64 / 1e3_f64;
    eval_number(tfa.ec, v)
}
