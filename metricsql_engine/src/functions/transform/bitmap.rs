use crate::functions::arg_parse::get_float_arg;
use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{RuntimeResult, Timeseries};

pub(crate) fn transform_bitmap_and(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_and)
}

pub(crate) fn transform_bitmap_or(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_or)
}

pub(crate) fn transform_bitmap_xor(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_bitmap_impl(tfa, bitmap_xor)
}

pub(crate) fn transform_bitmap_impl(
    tfa: &mut TransformFuncArg,
    bitmap_func: fn(a: u64, b: u64) -> u64,
) -> RuntimeResult<Vec<Timeseries>> {
    let mask = get_float_arg(&tfa.args, 1, None)? as u64;

    let tf = |values: &mut [f64]| {
        for v in values.iter_mut() {
            if v.is_nan() {
                continue;
            }
            *v = bitmap_func(*v as u64, mask) as f64;
        }
    };

    transform_series(tfa, tf)
}

#[inline]
fn bitmap_and(a: u64, b: u64) -> u64 {
    a & b
}

#[inline]
fn bitmap_or(a: u64, b: u64) -> u64 {
    a | b
}

#[inline]
fn bitmap_xor(a: u64, b: u64) -> u64 {
    a ^ b
}
