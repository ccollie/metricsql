use std::collections::HashSet;

use crate::eval::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::signature::Signature;
use crate::{EvalConfig, QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn handle_union(
    args: Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if args.len() < 1 {
        return eval_number(ec, f64::NAN);
    }

    let len = args[0].len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);
    let mut m: HashSet<Signature> = HashSet::with_capacity(len);

    fn process_vector(
        v: &mut Vec<Timeseries>,
        m: &mut HashSet<Signature>,
        rvs: &mut Vec<Timeseries>,
    ) {
        for mut ts in v.iter_mut() {
            ts.metric_name.sort_tags();
            let key = ts.metric_name.signature();
            if m.insert(key) {
                rvs.push(std::mem::take(&mut ts));
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

pub(crate) fn transform_union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // we don't use args after this
    let args = std::mem::take(&mut tfa.args);
    handle_union(args, &mut tfa.ec)
}
