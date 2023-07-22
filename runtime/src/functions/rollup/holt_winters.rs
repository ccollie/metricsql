use crate::functions::rollup::{RollupFuncArg, RollupHandlerEnum};
use crate::functions::types::get_scalar_param_value;
use crate::{EvalConfig, QueryValue, RuntimeResult};

pub(super) fn new_rollup_holt_winters(
    args: &Vec<QueryValue>,
    ec: &EvalConfig,
) -> RuntimeResult<RollupHandlerEnum> {
    // unwrap is sound since arguments are checked before this is called
    let sf = get_scalar_param_value(args, 1, "holt_winters", "sf")?;
    let tf = get_scalar_param_value(args, 2, "holt_winters", "tf")?;

    Ok(RollupHandlerEnum::General(Box::new(
        move |rfa: &mut RollupFuncArg| -> f64 { holt_winters_internal(rfa, sf, tf) },
    )))
}

fn holt_winters_internal(rfa: &mut RollupFuncArg, sf: f64, tf: f64) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup fns.
    if rfa.values.is_empty() {
        return rfa.prev_value;
    }

    if sf <= 0.0 || sf >= 1.0 {
        return f64::NAN;
    }

    if tf <= 0.0 || tf >= 1.0 {
        return f64::NAN;
    }

    let mut ofs = 0;

    // See https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing .
    // TODO: determine whether this shit really works.
    let mut s0 = rfa.prev_value;
    if s0.is_nan() {
        ofs = 1;
        s0 = rfa.values[0];

        if rfa.values.len() <= 1 {
            return s0;
        }
    }

    let mut b0 = rfa.values[ofs] - s0;
    for v in rfa.values[ofs..].iter() {
        let s1 = sf * v + (1.0 - sf) * (s0 + b0);
        let b1 = tf * (s1 - s0) + (1.0 - tf) * b0;
        s0 = s1;
        b0 = b1
    }

    s0
}
