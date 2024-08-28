use crate::functions::arg_parse::get_scalar_arg_as_vec;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn clamp(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let min_values = get_scalar_arg_as_vec(&tfa.args, 1, tfa.ec)?;
    let max_values = get_scalar_arg_as_vec(&tfa.args, 2, tfa.ec)?;
    // todo: are these guaranteed to be of equal length ?
    let tf = |values: &mut [f64]| {
        for ((v, min), max) in values
            .iter_mut()
            .zip(min_values.iter())
            .zip(max_values.iter())
        {
            if *v < *min {
                *v = *min;
            } else if *v > *max {
                *v = *max;
            }
        }
    };

    transform_series(tfa, tf)
}

pub(crate) fn clamp_max(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    process_values("clamp_max", tfa, |x: &mut f64, compare_to: f64| {
        if *x > compare_to {
            *x = compare_to;
        }
    })
}

pub(crate) fn clamp_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    process_values("clamp_min", tfa, |x: &mut f64, compare_to: f64| {
        if *x < compare_to {
            *x = compare_to;
        }
    })
}

fn process_values(
    name: &'static str,
    tfa: &mut TransformFuncArg,
    f: fn(x: &mut f64, compare_to: f64),
) -> RuntimeResult<Vec<Timeseries>> {
    // todo: check bounds
    let arg = tfa.args.get(1);
    if arg.is_none() {
        let msg = format!("{name}() : missing scalar arg # {}", 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    let arg = arg.unwrap();
    match arg {
        QueryValue::Scalar(val) => {
            let val = *val;
            let tf = |values: &mut [f64]| {
                for v in values.iter_mut() {
                    f(v, val);
                }
            };
            transform_series(tfa, tf)
        }
        QueryValue::InstantVector(s) => {
            if s.len() != 1 {
                let msg = format!(
                    "{name}() : arg # 1 must contain a single timeseries; got {} timeseries",
                    s.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            let comparison_values = &s[0].values.clone();
            let tf = |values: &mut [f64]| {
                for (v, val) in values.iter_mut().zip(comparison_values.iter()) {
                    f(v, *val);
                }
            };
            transform_series(tfa, tf)
        }
        _ => {
            let msg = format!(
                "{name}() arg #1 expected float or a single timeseries; got {}",
                arg.data_type()
            );
            Err(RuntimeError::ArgumentError(msg))
        }
    }
}
