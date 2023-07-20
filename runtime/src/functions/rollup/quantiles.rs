use crate::functions::rollup::{RollupFuncArg, RollupHandlerEnum};
use crate::functions::types::{get_scalar_param_value, get_string_param_value};
use crate::{QueryValue, RuntimeResult};
use lib::get_float64s;
use std::cmp::Ordering;
use std::ops::DerefMut;
use tinyvec::TinyVec;

/// quantiles calculates the given phis from originValues without modifying originValues, appends
/// them to qs and returns the result.
pub(crate) fn quantiles(qs: &mut [f64], phis: &[f64], origin_values: &[f64]) {
    if origin_values.len() <= 64 {
        let mut vec = tiny_vec!([f64; 64]);
        prepare_tv_for_quantile_float64(&mut vec, origin_values);
        return quantiles_sorted(qs, phis, &vec);
    }

    let mut block = get_float64s(phis.len());
    let a = block.deref_mut();
    prepare_for_quantile_float64(a, origin_values);
    quantiles_sorted(qs, phis, a)
}

/// calculates the given phi from origin_values without modifying origin_values
pub(crate) fn quantile(phi: f64, origin_values: &[f64]) -> f64 {
    // todo: smallvec
    let mut block = get_float64s(origin_values.len());
    prepare_for_quantile_float64(&mut block, origin_values);
    quantile_sorted(phi, &block)
}

/// prepare_for_quantile_float64 copies items from src to dst but removes NaNs and sorts the dst
fn prepare_for_quantile_float64(dst: &mut Vec<f64>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue;
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// copies items from src to dst but removes NaNs and sorts the dst
fn prepare_tv_for_quantile_float64(dst: &mut TinyVec<[f64; 64]>, src: &[f64]) {
    for v in src {
        if v.is_nan() {
            continue;
        }
        dst.push(*v);
    }
    dst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));
}

/// calculates the given phis over a sorted list of values, appends them to qs and returns the result.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub(crate) fn quantiles_sorted(qs: &mut [f64], phis: &[f64], values: &[f64]) {
    for (phi, qs) in phis.iter().zip(qs.iter_mut()) {
        *qs = quantile_sorted(*phi, values);
    }
}

/// quantile_sorted calculates the given quantile over a sorted list of values.
///
/// It is expected that values won't contain NaN items.
/// The implementation mimics Prometheus implementation for compatibility's sake.
pub(crate) fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
    if values.len() == 0 || phi.is_nan() {
        return f64::NAN;
    }
    if phi < 0.0 {
        return f64::NEG_INFINITY;
    }
    if phi > 1.0 {
        return f64::INFINITY;
    }
    let n = values.len();
    let rank = phi * (n - 1) as f64;

    let lower_index = std::cmp::max(0, rank.floor() as usize);
    let upper_index = std::cmp::min(n - 1, lower_index + 1) as usize;

    let weight = rank - rank.floor();
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
}

pub(super) fn new_rollup_quantiles(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi_label = get_string_param_value(&args, 0, "quantiles", "phi_label").unwrap();
    let cap = args.len() - 1;

    let mut phis = Vec::with_capacity(cap);
    // todo: smallvec ??
    let mut phi_labels: Vec<String> = Vec::with_capacity(cap);

    for i in 1..args.len() {
        // unwrap should be safe, since parameters types are checked before calling the function
        let v = get_scalar_param_value(args, i, "quantiles", "phi").unwrap();
        phis.push(v);
        phi_labels.push(format!("{}", v));
    }

    let f = Box::new(move |rfa: &mut RollupFuncArg| -> f64 {
        quantiles_impl(rfa, &phi_label, &phis, &phi_labels)
    });

    Ok(RollupHandlerEnum::General(f))
}

pub(super) fn new_rollup_quantile(args: &Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum> {
    let phi = get_scalar_param_value(args, 0, "quantile_over_time", "phi")?;

    let rf = Box::new(move |rfa: &mut RollupFuncArg| {
        // There is no need in handling NaNs here, since they must be cleaned up
        // before calling rollup fns.
        quantile(phi, &rfa.values)
    });

    Ok(RollupHandlerEnum::General(rf))
}

fn quantiles_impl(
    rfa: &mut RollupFuncArg,
    label: &str,
    phis: &Vec<f64>,
    phi_labels: &Vec<String>,
) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return rfa.prev_value;
    }
    if rfa.values.len() == 1 {
        // Fast path - only a single value.
        return rfa.values[0];
    }
    // tinyvec ?
    let mut qs = get_float64s(phis.len());
    quantiles(qs.deref_mut(), &phis, &rfa.values);
    let idx = rfa.idx;
    let tsm = rfa.tsm.as_ref().unwrap();
    let mut wrapped = tsm.borrow_mut();
    for (phi_str, quantile) in phi_labels.iter().zip(qs.iter()) {
        let ts = wrapped.get_or_create_timeseries(label, phi_str);
        ts.values[idx] = *quantile;
    }

    return f64::NAN;
}
