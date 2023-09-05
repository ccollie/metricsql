use std::collections::HashSet;

use crate::execution::{eval_number, EvalConfig};
use crate::functions::transform::TransformFuncArg;
use crate::signature::Signature;
use crate::{QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // we don't use args after this
    let args = std::mem::take(&mut tfa.args);
    handle_union(args, tfa.ec)
}

pub(crate) fn handle_union(
    args: Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if args.is_empty() {
        return eval_number(ec, f64::NAN);
    }

    let len = args[0].len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);
    let mut m: HashSet<Signature> = HashSet::with_capacity(len);

    fn process_vector(v: &mut [Timeseries], m: &mut HashSet<Signature>, rvs: &mut Vec<Timeseries>) {
        for ts in v.iter_mut() {
            let key = ts.metric_name.signature();
            if m.insert(key) {
                rvs.push(std::mem::take(ts));
            }
        }
    }

    let mut args = args;
    for arg in args.iter_mut() {
        // done this way to avoid allocating a new vector in the case of a InstantVector
        match arg {
            QueryValue::Scalar(v) => {
                let mut ts = eval_number(ec, *v)?;
                process_vector(&mut ts, &mut m, &mut rvs);
            }
            QueryValue::InstantVector(v) => process_vector(v, &mut m, &mut rvs),
            QueryValue::RangeVector(v) => process_vector(v, &mut m, &mut rvs),
            _ => {
                return Err(RuntimeError::ArgumentError(
                    "expected instant or range vector".to_string(),
                ));
            }
        }
    }

    Ok(rvs)
}
