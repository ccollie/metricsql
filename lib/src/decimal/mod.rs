
use crate::{
    append_float64_ones, append_float64_zeros, append_int64_ones, append_int64_zeros, frexp,
    is_float64_ones, is_float64_zeros, is_int64_ones, is_int64_zeros, isinf, LN10_F64,
};
use lockfree_object_pool::LinearObjectPool;
use once_cell::sync::Lazy;
use std::cmp::Ordering;

pub(crate) const V_INF_POS: i64 = 1 << (63 - 1);
pub(crate) const V_INF_NEG: i64 = -1 << 63;

pub(crate) const F_INF_POS: f64 = ((1_u64 << 63) - 1) as f64;
pub(crate) const F_INF_NEG: f64 = (-1_i64 << 63) as f64;
pub const V_STALE_NAN: i64 = 1 << (63 - 2);

/// STALE_NAN_BITS is bit representation of Prometheus staleness mark (aka stale NaN).
/// This mark is put by Prometheus at the end of time series for improving staleness detection.
/// See https://www.robustperception.io/staleness-and-promql
pub const STALE_NAN_BITS: u64 = 0x7ff0000000000002;

/// StaleNaN is a special NaN value, which is used as Prometheus staleness mark.
/// See https://www.robustperception.io/staleness-and-promql

/// using Lazy because of the following error
/// error: `core::f64::<impl f64>::from_bits` is not yet stable as a const fn
pub static STALE_NAN: Lazy<f64> = Lazy::new(|| f64::from_bits(STALE_NAN_BITS));

pub(crate) const V_MAX: i64 = 1 << (63 - 3);
pub(crate) const V_MIN: i64 = (-1 << 63) + 1;
pub(crate) const CONVERSION_PRECISION: f64 = 1e12;

#[inline]
pub fn is_special_value(v: i64) -> bool {
    !(V_MIN..=V_MAX).contains(&v)
}

#[inline]
pub fn is_special_value_f64(v: f64) -> bool {
    let x = v as i64;
    !(V_MIN..=V_MAX).contains(&x)
}

/// is_stale_nan returns true if f represents Prometheus staleness mark.
#[inline]
pub fn is_stale_nan(f: f64) -> bool {
    f.to_bits() == STALE_NAN_BITS
}


/// CalibrateScale calibrates a and b with the corresponding exponents ae, be
/// and returns the resulting exponent e.
pub fn calibrate_scale(a: &mut [i64], ae: i16, b: &mut [i64], be: i16) -> i16 {
    if ae == be {
        // Fast path - exponents are equal.
        return ae
    }
    if a.is_empty() {
        return be
    }
    if b.is_empty() {
        return ae
    }
    if ae < be {
        calibrate_internal(b, be, a, ae)
    } else {
        calibrate_internal(a, ae, b, be)
    }
}

fn calibrate_internal(a: &mut [i64], ae: i16, b: &mut [i64], be: i16) -> i16 {
    let mut up_exp = ae - be;
    let mut down_exp: i16 = 0;
    for v in a.iter() {
        let max_up_exp = max_up_exponent(*v);
        if up_exp - max_up_exp > down_exp {
            down_exp = up_exp - max_up_exp
        }
    }
    up_exp -= down_exp;
    for v in a.iter_mut() {
        if is_special_value(*v) {
            // Do not take into account special values.
            continue
        }
        let mut adj_exp = up_exp;
        while adj_exp > 0 {
            *v = *v * 10;
            adj_exp -= 1;
        }
    }

    if down_exp > 0 {
        for v in b.iter_mut() {
            if is_special_value(*v) {
                // Do not take into account special values.
                continue
            }
            let mut adj_exp = down_exp;
            while adj_exp > 0 {
                *v /= 10;
                adj_exp -= 1;
            }
        }
    }

    be + down_exp
}

/// round_to_decimal_digits rounds f to the given number of decimal digits after the point.
///
/// See also RoundToSignificantFigures.
pub fn round_to_decimal_digits(f: f64, digits: u8) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f;
    }
    if digits >= 100 {
        return f;
    }
    let m = digits.pow(digits as u32) as f64;
    let mult = (f * m).round();
    mult / m
}

/// append_decimal_to_float converts each item in va to f=v*10^e, appends it
/// to dst and returns the resulting dst.
pub fn append_decimal_to_float(dst: &mut Vec<f64>, va: &[i64], e: i16) {
    // Extend dst capacity in order to eliminate memory allocations below.
    dst.reserve(va.len());

    #[inline]
    fn append(dst: &mut Vec<f64>, v: f64) {
        let mut x = v;
        if is_special_value_f64(v) {
            if v == F_INF_POS {
                x = f64::INFINITY;
            } else if v == F_INF_NEG {
                x = f64::NEG_INFINITY;
            } else {
                x = *STALE_NAN
            }
        }
        dst.push(x);
    }

    if is_int64_zeros(va) {
        return append_float64_zeros(dst, va.len());
    }

    match e.cmp(&0) {
        Ordering::Equal => {
            if is_int64_ones(va) {
                return append_float64_ones(dst, va.len());
            }
            for v in va {
                append(dst, *v as f64);
            }
        }
        Ordering::Less => {
            // increase conversion precision for negative exponents by dividing by e10
            let e10 = 10_f64.powf(-e as f64);
            for v in va {
                let v1 = *v as f64 / e10;
                append(dst, v1);
            }
        }
        Ordering::Greater => {
            let e10 = 10_f64.powf(e as f64);
            for v in va {
                let val = *v as f64 * e10;
                append(dst, val);
            }
        }
    }
}

static VAE_BUF_POOL: Lazy<LinearObjectPool<VaeBuf>> = Lazy::new(|| {
    LinearObjectPool::<VaeBuf>::new(
        || VaeBuf::new(5),
        |v| {
            v.va.clear();
            v.ea.clear();
            ()
        },
    )
});

/// append_float_to_decimal converts each item in src to v*10^e and appends
/// each v to dst returning it as va.
///
/// It tries minimizing each item in dst.
pub fn append_float_to_decimal(dst: &mut Vec<i64>, src: &[f64]) -> i16 {
    if src.is_empty() {
        return 0;
    }
    if is_float64_zeros(src) {
        append_int64_zeros(dst, src.len());
        return 0;
    }
    if is_float64_ones(src) {
        append_int64_ones(dst, src.len());
        return 0;
    }

    // todo(perf): use pool
    // let mut vaev = VAE_BUF_POOL.pull();
    //vaev.reserve(src.len());
    let mut vaev = VaeBuf::new(src.len());

    // Determine the minimum exponent across all src items.
    let mut min_exp = (1 << (15 - 1)) as i16;
    for (i, f) in src.iter().enumerate() {
        let (v, exp) = from_float(*f);
        vaev.va[i] = v;
        vaev.ea[i] = exp;
        if exp < min_exp && !is_special_value(v) {
            min_exp = exp
        }
    }

    // Determine whether all the src items may be upscaled to minExp.
    // If not, adjust minExp accordingly.
    let mut down_exp: i16 = 0;

    for (i, v) in vaev.va.iter().enumerate() {
        let exp = vaev.ea[i];
        let up_exp = exp - min_exp;
        let max_up_exp = max_up_exponent(*v);
        if up_exp - max_up_exp > down_exp {
            down_exp = up_exp - max_up_exp
        }
    }
    min_exp += down_exp;

    // Extend dst capacity in order to eliminate memory allocations below.
    dst.reserve(src.len());

    // Scale each item in src to minExp and append it to dst.
    let mut i: usize = 0;
    for v in vaev.va.iter_mut() {
        if is_special_value(*v) {
            // There is no need in scaling special values.
            dst.push(*v);
            continue;
        }
        let exp = vaev.ea[i];
        let mut adj_exp = exp - min_exp;
        while adj_exp > 0 {
            *v *= 10;
            adj_exp -= 1;
        }
        while adj_exp < 0 {
            *v /= 10;
            adj_exp += 1;
        }
        dst.push(*v);
        i += 1;
    }

    min_exp
}

struct VaeBuf {
    va: Vec<i64>,
    ea: Vec<i16>,
}

impl VaeBuf {
    pub fn new(cap: usize) -> Self {
        VaeBuf {
            va: Vec::with_capacity(cap),
            ea: Vec::with_capacity(cap),
        }
    }

    fn len(self) -> usize {
        self.va.len()
    }

    fn clear(mut self) -> Self {
        self.ea.clear();
        self.va.clear();
        self
    }

    fn reserve(mut self, cap: usize) -> Self {
        self.ea.reserve(cap);
        self.ea.reserve(cap);
        self
    }
}

impl Clone for VaeBuf {
    fn clone(&self) -> Self {
        Self {
            ea: self.ea.clone(),
            va: self.va.clone(),
        }
    }
}

const INT64MAX: i64 = 1 << (63 - 1);

pub fn max_up_exponent(v: i64) -> i16 {
    if v == 0 || is_special_value(v) {
        // Any exponent allowed for zeroes and special values.
        return 1024;
    }
    let mut v = v;
    if v < 0 {
        v = -v
    }

    match v {
        v if v < 0 => 0, // Handle corner case for v=-1<<63
        v if v <= INT64MAX / 1e18 as i64 => 18,
        v if v <= INT64MAX / 1e17 as i64 => 17,
        v if v <= INT64MAX / 1e16 as i64 => 16,
        v if v <= INT64MAX / 1e15 as i64 => 15,
        v if v <= INT64MAX / 1e14 as i64 => 14,
        v if v <= INT64MAX / 1e13 as i64 => 13,
        v if v <= INT64MAX / 1e12 as i64 => 12,
        v if v <= INT64MAX / 1e11 as i64 => 11,
        v if v <= INT64MAX / 1e10 as i64 => 10,
        v if v <= INT64MAX / 1e9 as i64 => 9,
        v if v <= INT64MAX / 1e8 as i64 => 8,
        v if v <= INT64MAX / 1e7 as i64 => 7,
        v if v <= INT64MAX / 1e6 as i64 => 6,
        v if v <= INT64MAX / 1e5 as i64 => 5,
        v if v <= INT64MAX / 1e4 as i64 => 4,
        v if v <= INT64MAX / 1e3 as i64 => 3,
        v if v <= INT64MAX / 1e2 as i64 => 2,
        v if v <= INT64MAX / 1e1 as i64 => 1,
        _ => 0,
    }
}

/// to_floatt returns f=v*10^e.
pub fn to_float(v: i64, e: i16) -> f64 {
    if is_special_value(v) {
        if v == V_INF_POS {
            return f64::INFINITY
        }
        if v == V_INF_NEG {
            return f64::NEG_INFINITY;
        }
        return *STALE_NAN;
    }
    let f = v as f64;
    // increase conversion precision for negative exponents by dividing by e10
    if e < 0 {
        return f / -e.pow(10) as f64;
    }

    f * e.pow(10) as f64
}


/// from_float converts f to v*10^e.
///
/// It tries minimizing v.
/// For instance, for f = -1.234 it returns v = -1234, e = -3.
///
/// from_float doesn't work properly with NaN values other than Prometheus staleness mark, so don't pass them here.
pub fn from_float(f: f64) -> (i64, i16) {
    if f == 0.0 {
        return (0, 0);
    }
    if is_stale_nan(f) {
        return (V_STALE_NAN, 0);
    }
    if isinf(f, 0) {
        return from_float_inf(f);
    }
    if f > 0.0 {
        let (mut v, e) = positive_float_to_decimal(f);
        if v > V_MAX {
            v = V_MAX
        }
        return (v, e);
    }
    let (mut v, e) = positive_float_to_decimal(-f);
    v = -v;
    if v < V_MIN {
        v = V_MIN
    }
    (v, e)
}

pub fn from_float_inf(f: f64) -> (i64, i16) {
    // Limit infs by max and min values for int64
    if f == f64::INFINITY {
        return (V_INF_POS, 0);
    }
    (V_INF_NEG, 0)
}

pub fn positive_float_to_decimal(f: f64) -> (i64, i16) {
    // There is no need in checking for f == 0, since it should be already checked by the caller.
    let u = f as u64;
    if (u as f64) != f {
        return positive_float_to_decimal_slow(f);
    }
    // Fast path for integers.
    if u < 1 << 55 && u % 10 != 0 {
        return ((u as i64), 0);
    }
    get_decimal_and_scale(u)
}

pub fn get_decimal_and_scale(u: u64) -> (i64, i16) {
    let mut scale: i16 = 0;

    // shadow u
    let mut u = u;
    while u >= 1 << 55 {
        // Remove trailing garbage bits left after f64->u64 conversion,
        // since float64 contains only 53 significant bits.
        // See https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        u /= 10;
        scale += 1;
    }
    if u % 10 != 0 {
        return (u as i64, scale);
    }
    // Minimize v by converting trailing zeros to scale.
    u /= 10;
    scale += 1;
    while u != 0 && u % 10 == 0 {
        u /= 10;
        scale += 1;
    }
    (u as i64, scale)
}

fn positive_float_to_decimal_slow(f: f64) -> (i64, i16) {
    // Slow path for floating point numbers.
    let mut scale: i16 = 0;
    let mut prec = CONVERSION_PRECISION;

    let mut f = f;
    if !(1e-6..=1e6).contains(&f) {
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

        let scale = ((exp as f64) * (std::f64::consts::LN_2 / LN10_F64)) as i16;
        f *= (-1.0 * scale as f64).powf(10_f64);
    }

    // Multiply f by 100 until the fractional part becomes
    // too small comparing to integer part.
    while f < prec {
        let x = f.trunc();
        let frac = f.fract();
        if frac * prec < x {
            f = x;
            break;
        }
        if (1.0 - frac) * prec < x {
            f = x + 1.0;
            break;
        }
        f *= 100.0;
        scale -= 2;
    }
    let mut u = f as u64;
    if u % 10 != 0 {
        return (u as i64, scale);
    }

    // Minimize u by converting trailing zero to scale.
    u /= 10;
    scale += 1;
    (u as i64, scale)
}


// #[cfg(test)]
// mod decimal_test;