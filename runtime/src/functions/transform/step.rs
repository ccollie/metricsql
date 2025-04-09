use crate::execution::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeResult};

pub(crate) fn step(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let v = tfa.ec.step.as_secs() as f64;
    eval_number(tfa.ec, v)
}
