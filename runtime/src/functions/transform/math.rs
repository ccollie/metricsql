use num_traits::FloatConst;

use lib::{copysign, fmod, from_float, modf};

use crate::eval::eval_number;
use crate::functions::arg_parse::get_scalar_arg_as_vec;
use crate::functions::transform::transform_fns::{transform_series, TransformFuncArg};
use crate::{RuntimeError, RuntimeResult, Timeseries};

macro_rules! math_fn {
    ($name: ident, $func: expr) => {
        pub(crate) fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            math_func_impl(tfa, $func)
        }
    };
}

fn math_func_impl(
    tfa: &mut TransformFuncArg,
    op: fn(f64) -> f64,
) -> RuntimeResult<Vec<Timeseries>> {
    let tfe = |values: &mut [f64]| {
        for value in values.iter_mut() {
            *value = op(*value)
        }
    };
    transform_series(tfa, tfe)
}

math_fn!(transform_abs, |x: f64| x.abs());
math_fn!(transform_acos, |x: f64| x.acos());
math_fn!(transform_acosh, |x: f64| x.acosh());
math_fn!(transform_asin, |x: f64| x.asin());
math_fn!(transform_asinh, |x: f64| x.asinh());
math_fn!(transform_atan, |x: f64| x.atan());
math_fn!(transform_atanh, |x: f64| x.atanh());
math_fn!(transform_ceil, |x: f64| x.ceil());
math_fn!(transform_cos, |x: f64| x.cos());
math_fn!(transform_cosh, |x: f64| x.cosh());
math_fn!(transform_deg, |x: f64| x.to_degrees());
math_fn!(transform_exp, |x: f64| x.exp());
math_fn!(transform_floor, |x: f64| x.floor());
math_fn!(transform_ln, |x: f64| x.ln());
math_fn!(transform_log2, |x: f64| x.log2());
math_fn!(transform_log10, |x: f64| x.log10());
math_fn!(transform_rad, |x: f64| x.to_radians());
math_fn!(transform_sin, |x: f64| x.sin());
math_fn!(transform_sinh, |x: f64| x.sinh());
math_fn!(transform_sqrt, |x: f64| x.sqrt());
math_fn!(transform_tan, |x: f64| x.tan());
math_fn!(transform_tanh, |x: f64| x.tanh());

pub(crate) fn transform_round(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
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
        let mut n_prev: f64 = values[0];
        let mut p10: f64 = 0.0;
        for (v, n) in values.iter_mut().zip(nearest.iter()) {
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
    };

    transform_series(tfa, tf)
}

pub(crate) fn transform_pi(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_number(&tfa.ec, f64::PI())
}

pub(crate) fn transform_sgn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        for v in values {
            *v = v.signum();
        }
    };

    transform_series(tfa, tf)
}
