use crate::execution::{eval_number, EvalConfig};
use crate::functions::transform::TransformFuncArg;
use crate::types::{FunctionArgs, QueryValue, Timeseries};
use crate::{RuntimeError, RuntimeResult};
use metricsql_common::prelude::SignatureSet;
use crate::functions::utils::are_all_args_scalar;

pub(crate) fn union(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args = std::mem::take(&mut tfa.args);
    handle_union(args, tfa.ec)
}

pub(crate) fn handle_union(
    args: FunctionArgs,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if args.is_empty() {
        return eval_number(ec, f64::NAN);
    }

    let len = args[0].len();
    let mut rvs: Vec<Timeseries> = Vec::with_capacity(len);

    if are_all_args_scalar(&args) {
        // Special case for (v1,...,vN) where vX are scalars - return all the scalars as time series.
        // This is needed for "q == (v1,...,vN)" and "q != (v1,...,vN)" cases, where vX are numeric constants.
        for arg in args.into_iter() {
            match arg {
                QueryValue::Scalar(v) => {
                    let mut ts = eval_number(ec, v)?;
                    rvs.append(&mut ts);
                }
                QueryValue::InstantVector(v) => {
                    let mut ts = v;
                    rvs.append(&mut ts);
                }
                _ => {
                    return Err(RuntimeError::ArgumentError("expected scalar".to_string()));
                }
            }
        }
        return Ok(rvs);
    }

    let mut set = SignatureSet::new();

    fn process_vector(v: Vec<Timeseries>, m: &mut SignatureSet, rvs: &mut Vec<Timeseries>) {
        for ts in v.into_iter() {
            let key = ts.metric_name.signature();
            if m.insert(key) {
                rvs.push(ts);
            }
        }
    }

    for arg in args.into_iter() {
        // done this way to avoid allocating a new vector in the case of an InstantVector
        match arg {
            QueryValue::Scalar(v) => {
                let ts = eval_number(ec, v)?;
                process_vector(ts, &mut set, &mut rvs);
            }
            QueryValue::InstantVector(v) => process_vector(v, &mut set, &mut rvs),
            QueryValue::RangeVector(v) => process_vector(v, &mut set, &mut rvs),
            _ => {
                return Err(RuntimeError::ArgumentError(
                    format!("Unexpected argument in \"union()\": expected instant or range vector, got: {arg}"),
                ));
            }
        }
    }

    Ok(rvs)
}