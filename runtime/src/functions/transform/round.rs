use lib::{copysign, fmod, from_float, modf};

use crate::functions::arg_parse::get_scalar_arg_as_vec;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn round(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args_len = tfa.args.len();

    if args_len < 1 || args_len > 2 {
        return Err(RuntimeError::ArgumentError(format!(
            "unexpected number of arguments: #{}; want 1 or 2",
            tfa.args.len()
        )));
    }

    let nearest = if args_len == 1 {
        let len = tfa.ec.data_points();
        vec![1_f64; len]
    } else {
        get_scalar_arg_as_vec(&tfa.args, 1, tfa.ec)?
    };

    let tf = move |values: &mut [f64]| {
        prometheus_round(values, &nearest);
    };

    transform_series(tfa, tf)
}

fn orig_round(values: &mut [f64], nearest: &[f64]) {
    let mut n_prev: f64 = values[0];
    let mut p10: f64 = 0.0;
    for (v, n) in values.iter_mut().zip(nearest) {
        if *n != n_prev {
            n_prev = *n;
            let (_, e) = from_float(*n);
            p10 = -(e as f64).powi(10);
        }
        *v += 0.5 * copysign(*n, *v);
        *v -= fmod(*v, *n);
        let (x, _) = modf(*v * p10);
        *v = x / p10;
    }
}

fn prometheus_round(vals: &mut [f64], nearest: &[f64]) {
    // round returns a number rounded to to_nearest.
    // Ties are solved by rounding up.
    if nearest.len() == 1 {
        let to_nearest = nearest[0];
        // Invert as it seems to cause fewer floating point accuracy issues.
        let to_nearest_inverse = 1.0 / to_nearest;

        for v in vals.iter_mut() {
            let f = (*v * to_nearest_inverse + 0.5).floor() / to_nearest_inverse;
            *v = f;
        }
    } else {
        for (v, n) in vals.iter_mut().zip(nearest) {
            // Invert as it seems to cause fewer floating point accuracy issues.
            let to_nearest_inverse = 1.0 / n;
            let f = (*v * to_nearest_inverse + 0.5).floor() / to_nearest_inverse;
            *v = f;
        }
    }
}
