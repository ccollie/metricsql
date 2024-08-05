use crate::{RuntimeResult, Timeseries};
use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;

#[inline]
pub(crate) fn transform_start(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let start = if tfa.ec.real_start == 0 {
        tfa.ec.start
    } else {
        tfa.ec.real_start
    };
    let v = start as f64 / 1e3_f64;
    eval_number(tfa.ec, v)
}
