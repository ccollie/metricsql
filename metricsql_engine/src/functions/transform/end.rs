use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

#[inline]
pub(crate) fn transform_end(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let end = if tfa.ec.real_end == 0 {
        tfa.ec.end
    } else {
        tfa.ec.real_end
    };
    let v = end as f64 / 1e3_f64;
    eval_number(tfa.ec, v)
}
