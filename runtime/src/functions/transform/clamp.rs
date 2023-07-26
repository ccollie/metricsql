use crate::functions::arg_parse::get_scalar_arg_as_vec;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{RuntimeResult, Timeseries};

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
    let max_values = get_scalar_arg_as_vec(&tfa.args, 1, tfa.ec)?;
    let tf = |values: &mut [f64]| {
        for (v, max) in values.iter_mut().zip(max_values.iter()) {
            if *v > *max {
                *v = *max;
            }
        }
    };

    transform_series(tfa, tf)
}

pub(crate) fn clamp_min(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let min_values = get_scalar_arg_as_vec(&tfa.args, 1, tfa.ec)?;
    let tf = |values: &mut [f64]| {
        for (v, min) in values.iter_mut().zip(min_values.iter()) {
            if *v < *min {
                *v = *min;
            }
        }
    };

    transform_series(tfa, tf)
}
