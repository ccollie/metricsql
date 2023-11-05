use crate::common::math::{quantiles, IQR_PHIS};
use crate::functions::rollup::RollupFuncArg;

pub(crate) fn rollup_outlier_iqr(rfa: &RollupFuncArg) -> f64 {
    // There is no need in handling NaNs here, since they must be cleaned up
    // before calling rollup funcs.

    // See Outliers section at https://en.wikipedia.org/wiki/Interquartile_range
    let values = rfa.values;
    if values.len() < 2 {
        return f64::NAN;
    }
    let mut qs = [0.0, 0.0];
    quantiles(&mut qs, &IQR_PHIS, values);
    let q25 = qs[0];
    let q75 = qs[1];
    let iqr = 1.5 * (q75 - q25);

    let v = values.last().unwrap();
    return if *v > q75 + iqr || *v < q25 - iqr {
        *v
    } else {
        f64::NAN
    };
}
