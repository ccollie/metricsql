use crate::eval::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeResult, Timeseries};

pub(crate) fn step(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.step as f64 / 1e3_f64;
    eval_number(&tfa.ec, v)
}
