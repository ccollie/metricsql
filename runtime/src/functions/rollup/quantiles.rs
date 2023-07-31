use std::ops::DerefMut;

use lib::get_pooled_vec_f64_filled;

use crate::common::math::{quantile, quantiles};
use crate::functions::rollup::{RollupFuncArg, RollupHandlerEnum};
use crate::functions::types::{get_scalar_param_value, get_string_param_value};
use crate::{EvalConfig, QueryValue, RuntimeResult};

pub(super) fn new_rollup_quantiles(
    args: &Vec<QueryValue>,
    _ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    let phi_label = get_string_param_value(args, 0, "quantiles", "phi_label").unwrap();
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

pub(super) fn new_rollup_quantile(
    args: &Vec<QueryValue>,
    _ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
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
    phis: &[f64],
    phi_labels: &[String],
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
    let mut qs = get_pooled_vec_f64_filled(phis.len(), 0f64);
    quantiles(qs.deref_mut(), phis, &rfa.values);
    let idx = rfa.idx;
    let tsm = rfa.tsm.as_ref().unwrap();
    let mut wrapped = tsm.borrow_mut();
    for (phi_str, quantile) in phi_labels.iter().zip(qs.iter()) {
        let ts = wrapped.get_or_create_timeseries(label, phi_str);
        ts.values[idx] = *quantile;
    }

    f64::NAN
}
