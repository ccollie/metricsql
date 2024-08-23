use std::sync::LazyLock;
use crate::decimal::utils::{frexp, modf};
use rand_distr::num_traits::{FloatConst, Zero};

const INT64MAX: u64 = (1<<63) - 1;

const EXPONENTS: [i64; 19] = [
    (INT64MAX / (1e18 as u64)) as i64,
    (INT64MAX / (1e17 as u64)) as i64,
    (INT64MAX / (1e16 as u64)) as i64,
    (INT64MAX / (1e15 as u64)) as i64,
    (INT64MAX / (1e14 as u64)) as i64,
    (INT64MAX / (1e13 as u64)) as i64,
    (INT64MAX / (1e12 as u64)) as i64,
    (INT64MAX / (1e11 as u64)) as i64,
    (INT64MAX / (1e10 as u64)) as i64,
    (INT64MAX / (1e9 as u64)) as i64,
    (INT64MAX / (1e8 as u64)) as i64,
    (INT64MAX / (1e7 as u64)) as i64,
    (INT64MAX / (1e6 as u64)) as i64,
    (INT64MAX / (1e5 as u64)) as i64,
    (INT64MAX / (1e4 as u64)) as i64,
    (INT64MAX / (1e3 as u64)) as i64,
    (INT64MAX / (1e2 as u64)) as i64,
    (INT64MAX / (1e1 as u64)) as i64,
    0,
];
fn max_up_exponent(v: i64) -> i16 {
    let mut v = v;
    if v == 0 || is_special_value(v) {
        // Any exponent allowed for zeroes and special values.
        return 1024
    }
    if v < 0 {
        v -= v;
    }
    if v < 0 {
        // Handle corner case for v=-1<<63
        return 0
    }
    if v <= (INT64MAX / (1e18 as u64)) as i64 {
        return 18
    }
    if v <= (INT64MAX / (1e17 as u64)) as i64 {
        return 17
    }
    if v <= (INT64MAX / (1e16 as u64)) as i64 {
        return 16
    }
    if v <= (INT64MAX / (1e15 as u64)) as i64 {
        return 15
    }
    if v <= (INT64MAX / (1e14 as u64)) as i64 {
        return 14
    }
    if v <= (INT64MAX / (1e13 as u64)) as i64 {
        return 13
    }
    if v <= (INT64MAX / (1e12 as u64)) as i64 {
        return 12
    }
    if v <= (INT64MAX / (1e11 as u64)) as i64 {
        return 11
    }
    if v <= (INT64MAX / (1e10 as u64)) as i64 {
        return 10
    }
    if v <= (INT64MAX / (1e9 as u64)) as i64 {
        return 9
    }
    if v <= (INT64MAX / (1e8 as u64)) as i64 {
        return 8
    }
    if v <= (INT64MAX / (1e7 as u64)) as i64 {
        return 7
    }
    if v <= (INT64MAX / (1e6 as u64)) as i64 {
        return 6
    }
    if v <= (INT64MAX / (1e5 as u64)) as i64 {
        return 5
    }
    if v <= (INT64MAX / (1e4 as u64)) as i64 {
        return 4
    }
    if v <= (INT64MAX / (1e3 as u64)) as i64 {
        return 3
    }
    if v <= (INT64MAX / (1e2 as u64)) as i64 {
        return 2
    }
    if v <= (INT64MAX / (1e1 as u64)) as i64 {
        return 1
    }
    0
}



const V_INF_POS: i64 = 9223372036854775807; // (1<<63) as i64 - 1;
const V_INF_NEG: i64 = -1 << 63;
const V_STALE_NAN: i64 = V_INF_POS - 1; // (1<<63) - 2;

const V_MAX: i64 = V_INF_POS - 2; // (1<<63) - 3;
const V_MIN: i64 = (-1<<63) + 1;

// STALE_NAN_BITS is bit representation of Prometheus staleness mark (aka stale NaN).
// This mark is put by Prometheus at the end of time series for improving staleness detection.
// See https://www.robustperception.io/staleness-and-promql
const STALE_NAN_BITS: u64 = 0x7ff0000000000002;

// StaleNaN is a special NaN value, which is used as Prometheus staleness mark.
// See https://www.robustperception.io/staleness-and-promql
const STALE_NAN: LazyLock<f64> = LazyLock::new(|| f64::from_bits(STALE_NAN_BITS));


// round_to_decimal_digits rounds f to the given number of decimal digits after the point.
//
// See also RoundToSignificantFigures.
pub fn round_to_decimal_digits(f: f64, digits: i32) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f
    }
    if digits <= -100 || digits >= 100 {
        return f
    }
    let scale: f64 = 10_f64.powi(digits);
    (f * scale).round() / scale
}

fn round(number: f64, rounding: i32) -> f64 {
    let scale: f64 = 10_f64.powi(rounding);
    (number * scale).round() / scale
}

// round_to_significant_figures rounds f to value with the given number of significant figures.
//
// See also round_to_decimal_digits.
pub fn round_to_significant_figures(f: f64, digits: i32) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f
    }
    if digits <= 0 || digits >= 18 {
        return f
    }
    if f.is_nan() || f.is_infinite() || f.is_zero() {
        return f
    }
    let n = digits.pow(10);
    let mut f = f;
    let is_negative = f.is_sign_negative();
    if is_negative {
        f = -f
    }
    let (mut v, mut e) = positive_float_to_decimal(f);
    v = v.max(V_MAX);

    let mut rem: i64 = 0;
    while v > n as i64 {
        rem = v % 10;
        v /= 10;
        e += 1;
    }
    if rem >= 5 {
        v += 1;
    }
    if is_negative {
        v -= v;
    }
    to_float(v, e)
}

const fn is_special_value(v: i64) -> bool {
    v > V_MAX || v < V_MIN
}

// is_stale_nan returns true if f represents Prometheus staleness mark.
pub fn is_stale_nan(f: f64) -> bool {
    f.to_bits() == STALE_NAN_BITS
}

const INF_POS: f64 = f64::INFINITY;
const INF_NEG: f64 = f64::NEG_INFINITY;


// to_float returns f=v*10^e.
fn to_float(v: i64, e: i16) -> f64 {
    if is_special_value(v) {
        if v == V_INF_POS {
            return INF_POS
        }
        if v == V_INF_NEG {
            return INF_NEG
        }
        return *STALE_NAN
    }
    let f = v as f64;
    // increase conversion precision for negative exponents by dividing by e10
    if e < 0 {
        return f / (-e).pow(10) as f64
    }
    f * e.pow(10) as f64
}


// from_float converts f to v*10^e.
//
// It tries minimizing v.
// For instance, for f = -1.234 it returns v = -1234, e = -3.
//
// from_float doesn't work properly with NaN values other than Prometheus staleness mark, so don't pass them here.
fn from_float(f: f64) -> (i64, i16) {
    if f == 0.0 {
        return (0, 0)
    }
    if is_stale_nan(f) {
        return (V_STALE_NAN, 0)
    }
    if f.is_infinite() {
        return from_float_inf(f);
    }
    if f > 0.0 {
        let (mut v, e) = positive_float_to_decimal(f);
        v = v.max(V_MAX);
        return (v, e)
    }
    let (mut v, e) = positive_float_to_decimal(-f);
    v -= v;
    v = v.min(V_MIN);
    (v, e)
}

fn from_float_inf(f: f64) -> (i64, i16) {
    // Limit infs by max and min values for int64
    if f.is_infinite() && f.is_sign_positive() {
        return (V_INF_POS, 0)
    }
    (V_INF_NEG, 0)
}

fn positive_float_to_decimal(f: f64) -> (i64, i16) {
    // There is no need in checking for f == 0, since it should be already checked by the caller.
    let u = f as u64;
    if (u as f64) != f {
        return positive_float_to_decimal_slow(f)
    }
    // Fast path for integers.
    if u < 1<<55 && u%10 != 0 {
        return (u as i64, 0)
    }
    get_decimal_and_scale(u)
}

fn get_decimal_and_scale(u: u64) -> (i64, i16) {
    let mut scale: i16 = 0;
    let mut u = u;
    while u >= 1<<55 {
        // Remove trailing garbage bits left after float64->uint64 conversion,
        // since float64 contains only 53 significant bits.
        // See https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        u /= 10;
        scale += 1;
    }
    if u%10 != 0 {
        return (u as i64, scale)
    }
    // Minimize v by converting trailing zeros to scale.
    u /= 10;
    scale += 1;
    while u != 0 && u%10 == 0 {
        u /= 10;
        scale += 1;
    }
    (u as i64, scale)
}

const CONVERSION_PRECISION: f64 = 1e12;

fn positive_float_to_decimal_slow(f: f64) -> (i64, i16) {
    // Slow path for floating point numbers.
    let mut scale: i16 = 0;
    let mut prec = CONVERSION_PRECISION;
    let mut f = f;
    if f > 1e6 || f < 1e-6 {
        // Normalize f, so it is in the small range suitable
        // for the next loop.
        if f > 1e6 {
            // Increase conversion precision for big numbers.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/213
            prec = 1e15
        }
        let (_, mut exp) = frexp(f);
        // Bound the exponent according to https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        // This fixes the issue https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1114
        if exp < -1022 {
            exp = -1022
        } else if exp > 1023 {
            exp = 1023
        }
        scale = ((exp as f64) * (f64::LN_2() / f64::LN_10())) as i16;
        f = f * (-scale).pow(10) as f64;
    }

    // Multiply f by 100 until the fractional part becomes
    // too small comparing to integer part.
    while f < prec {
        let (x, frac) = modf(f);
        if frac*prec < x {
            f = x;
            break
        }
        if (1f64-frac)*prec < x {
            f = x + 1f64;
            break
        }
        f *= 100f64;
        scale -= 2;
    }
    let mut u = f as u64;
    if u%10 != 0 {
        return (u as i64, scale)
    }

    // Minimize u by converting trailing zero to scale.
    u /= 10;
    scale += 1;
    (u as i64, scale)
}