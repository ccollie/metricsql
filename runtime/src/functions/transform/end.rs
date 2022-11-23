use crate::eval::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

#[inline]
pub(crate) fn transform_end(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.end as f64 / 1e3_f64;
    eval_number(&tfa.ec, v)
}
