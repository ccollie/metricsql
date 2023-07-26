use num_traits::FloatConst;

use lib::{copysign, fmod, from_float, modf};

use crate::eval::eval_number;
use crate::functions::arg_parse::{get_float_arg, get_scalar_arg_as_vec};
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{QueryValue, RuntimeError, RuntimeResult, Timeseries};

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

math_fn!(abs, |x: f64| x.abs());
math_fn!(acos, |x: f64| x.acos());
math_fn!(acosh, |x: f64| x.acosh());
math_fn!(asin, |x: f64| x.asin());
math_fn!(asinh, |x: f64| x.asinh());
math_fn!(atan, |x: f64| x.atan());
math_fn!(atanh, |x: f64| x.atanh());
math_fn!(ceil, |x: f64| x.ceil());
math_fn!(cos, |x: f64| x.cos());
math_fn!(cosh, |x: f64| x.cosh());
math_fn!(deg, |x: f64| x.to_degrees());
math_fn!(exp, |x: f64| x.exp());
math_fn!(floor, |x: f64| x.floor());
math_fn!(ln, |x: f64| x.ln());
math_fn!(log2, |x: f64| x.log2());
math_fn!(log10, |x: f64| x.log10());
math_fn!(rad, |x: f64| x.to_radians());
math_fn!(sin, |x: f64| x.sin());
math_fn!(sinh, |x: f64| x.sinh());
math_fn!(sqrt, |x: f64| x.sqrt());
math_fn!(tan, |x: f64| x.tan());
math_fn!(tanh, |x: f64| x.tanh());

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

// === round(Vector parser.ValueTypeVector, toNearest=1 Scalar) Vector ===
fn prometheus_round(vals: &mut Vec<f64>, args: &[QueryValue]) -> RuntimeResult<()> {
    // round returns a number rounded to to_nearest.
    // Ties are solved by rounding up.
    let mut to_nearest = 1_f64;
    if args.len() >= 2 {
        to_nearest = get_float_arg(args, 1, None)?;
    }
    // Invert as it seems to cause fewer floating point accuracy issues.
    let to_nearest_inverse = 1.0 / to_nearest;

    for v in vals.iter_mut() {
        let f = (*v * to_nearest_inverse + 0.5).floor() / to_nearest_inverse;
        *v = f;
    }

    Ok(())
}
