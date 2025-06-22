use num_traits::FloatConst;

use crate::execution::eval_number;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::types::Timeseries;
use crate::RuntimeResult;

macro_rules! math_fn {
    ($name: ident, $func: expr) => {
        pub(super) fn $name(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
            math_func_impl(tfa, $func)
        }
    };
}

math_fn!(abs, f64::abs);
math_fn!(acos, f64::acos);
math_fn!(acosh, f64::acosh);
math_fn!(asin, f64::asin);
math_fn!(asinh, f64::asinh);
math_fn!(atan, f64::atan);
math_fn!(atanh, f64::atanh);
math_fn!(ceil, f64::ceil);
math_fn!(cos, f64::cos);
math_fn!(cosh, f64::cosh);
math_fn!(deg, f64::to_degrees);
math_fn!(exp, f64::exp);
math_fn!(floor, f64::floor);
math_fn!(ln, f64::ln);
math_fn!(log2, f64::log2);
math_fn!(log10, f64::log10);
math_fn!(rad, f64::to_radians);
math_fn!(sin, f64::sin);
math_fn!(sinh, f64::sinh);
math_fn!(sqrt, f64::sqrt);
math_fn!(tan, f64::tan);
math_fn!(tanh, f64::tanh);

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

pub(crate) fn transform_pi(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    eval_number(tfa.ec, f64::PI())
}

pub(crate) fn sgn(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let tf = |values: &mut [f64]| {
        let zero = 0.0f64;
        for v in values {
            *v = match v.total_cmp(&zero) {
                std::cmp::Ordering::Less => -1.0f64,
                std::cmp::Ordering::Equal => 0.0f64,
                std::cmp::Ordering::Greater => 1.0f64,
            }
        }
    };

    transform_series(tfa, tf)
}
