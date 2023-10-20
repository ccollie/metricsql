use rand_distr::num_traits::Pow;

/// STALE_NAN_BITS is bit representation of Prometheus staleness mark (aka stale NaN).
/// This mark is put by Prometheus at the end of time series for improving staleness detection.
/// See https://www.robustperception.io/staleness-and-promql
/// StaleNaN is a special NaN value, which is used as Prometheus staleness mark.
pub const STALE_NAN_BITS: u64 = 0x7ff0000000000002;
pub(crate) const V_MAX: i64 = 1 << (63 - 3);
pub(crate) const V_MIN: i64 = (-1 << 63) + 1;

#[inline]
pub const fn is_special_value(v: i64) -> bool {
    if v < V_MIN || v > V_MAX {
        return true;
    }
    false
}

/// is_stale_nan returns true if f represents Prometheus staleness mark.
#[inline]
pub fn is_stale_nan(f: f64) -> bool {
    f.to_bits() == STALE_NAN_BITS
}

/// round_to_decimal_digits rounds f to the given number of decimal digits after the point.
///
/// See also RoundToSignificantFigures.
pub fn round_to_decimal_digits(f: f64, digits: i16) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f;
    }
    if digits <= -100 || digits >= 100 {
        return f;
    }
    let m = 10_f64.pow(digits);
    let mult = (f * m).round();
    mult / m
}

// #[cfg(test)]
// mod decimal_test;
