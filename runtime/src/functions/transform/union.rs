use crate::execution::{eval_number, EvalConfig};
use crate::functions::transform::TransformFuncArg;
use crate::types::{FunctionArgs, QueryValue, Timeseries};
use crate::{RuntimeError, RuntimeResult};
use metricsql_common::hash::{BuildNoHashHasher, Signature};
use metricsql_common::set::SmallSet;

pub(crate) fn union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    // we don't use args after this
    let mut args = std::mem::take(&mut tfa.args);
    handle_union(&mut args, tfa.ec)
}

type SignatureSet = SmallSet<16, Signature, BuildNoHashHasher<Signature>>;

pub(crate) fn handle_union(
    args: &mut FunctionArgs,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if args.is_empty() {
        return eval_number(ec, f64::NAN);
    }

    let len = args[0].len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);

    if are_all_args_scalar(args) {
        // Special case for (v1,...,vN) where vX are scalars - return all the scalars as time series.
        // This is needed for "q == (v1,...,vN)" and "q != (v1,...,vN)" cases, where vX are numeric constants.
        for arg in args.into_iter() {
            match arg {
                QueryValue::Scalar(v) => {
                    let mut ts = eval_number(ec, *v)?;
                    rvs.append(&mut ts);
                }
                QueryValue::InstantVector(ref mut v) => {
                    rvs.append(v);
                }
                _ => {
                    return Err(RuntimeError::ArgumentError("expected scalar".to_string()));
                }
            }
        }
        return Ok(rvs);
    }
    
    let mut set: SignatureSet = SmallSet::new();

    fn process_vector(v: &mut [Timeseries], m: &mut SignatureSet, rvs: &mut Vec<Timeseries>) {
        for ts in v.iter_mut() {
            let key = ts.metric_name.signature();
            if m.insert(key) {
                rvs.push(std::mem::take(ts));
            }
        }
    }

    for arg in args.iter_mut() {
        // done this way to avoid allocating a new vector in the case of an InstantVector
        match arg {
            QueryValue::Scalar(v) => {
                let mut ts = eval_number(ec, *v)?;
                process_vector(&mut ts, &mut set, &mut rvs);
            }
            QueryValue::InstantVector(v) => process_vector(v, &mut set, &mut rvs),
            QueryValue::RangeVector(v) => process_vector(v, &mut set, &mut rvs),
            _ => {
                return Err(RuntimeError::ArgumentError(
                    "expected instant or range vector".to_string(),
                ));
            }
        }
    }

    Ok(rvs)
}

fn are_all_args_scalar(args: &[QueryValue]) -> bool {
    args.iter().all(|arg| match arg {
        QueryValue::Scalar(_) => true,
        QueryValue::InstantVector(v) => {
            if v.len() != 1 {
                return false;
            }
            let mn = &v[0].metric_name;
            mn.is_empty()
        }
        _ => false,
    })
}
