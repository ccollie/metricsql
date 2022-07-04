// STALE_NAN_BITS is bit representation of Prometheus staleness mark (aka stale NaN).
// This mark is put by Prometheus at the end of time series for improving staleness detection.
// See https://www.robustperception.io/staleness-and-promql
const STALE_NAN_BITS: u64 = 0x7ff0000000000002;

pub(crate) const V_INF_POS: u64  = 1<<63 - 1;
pub(crate) const V_INF_NEG: i64  = -1 << 63;
pub(crate) const STALE_NAN: i64 = 1<<63 - 2;

pub(crate) const vMax: i64 = 1<<63 - 3;
pub(crate) const vMin: i64 = -1<<63 + 1;
pub(crate) const CONVERSION_PRECISION: f64 = 1e12;

pub fn is_special_value(v: i64) -> bool {
    return v > vMax || v < vMin
}

// IsStaleNaN returns true if f represents Prometheus staleness mark.
#[inline]
pub fn is_stale_nan(f: &f64) -> bool {
    f.to_bits() == staleNaNBits
}


// RoundToDecimalDigits rounds f to the given number of decimal digits after the point.
//
// See also RoundToSignificantFigures.
pub fn round_to_decimal_digits(f: &f64, digits: u8) -> f64 {
    if is_stale_nan(f) {
        // Do not modify stale nan mark value.
        return f;
    }
    if digits <= -100 || digits >= 100 {
        return f
    }
    let m = digits.pow(digits);
    let mult = ((f * m) as f64).round();
    return mult / m;
}

// append_decimal_to_float converts each item in va to f=v*10^e, appends it
// to dst and returns the resulting dst.
fn append_decimal_to_float(mut dst: &Vec<f64>, va: &[i64], e: i16) {
    let ofs = dst.len();
    // Extend dst capacity in order to eliminate memory allocations below.
    dst.reserve(va.len());

    #[inline]
    fn append(mut dst: &Vec<f64>, idx: usize, v: f64) {
        let mut x = v;
        if is_special_value(v) {
            if v == V_INF_POS {
                x = f64::INFINITY;
            } else if v == vInfNeg {
                x = f64::NEG_INFINITY;
            } else {
                x = STALE_NAN
            }
        }
        dst.push(x);
    }

    if fastnum.IsInt64Zeros(va) {
        return fastnum.AppendFloat64Zeros(dst, val.len())
    }

    if e == 0 {
        if fastnum.IsInt64Ones(va) {
            return fastnum.AppendFloat64Ones(dst, val.len())
        }
        for (i, v) in va.iter().enumerate() {
            append(dst, i + ofs, v as f64);
        }
    } else if e < 0 {
        // increase conversion precision for negative exponents by dividing by e10
        let e10 = math.Pow10(int(-e));
        for (i, v) in va.iter().enumerate() {
            let v1 = (v / e10);
            append(dst, i + ofs, v1);
        }
    } else {
        let e10 = math.Pow10(int(e));
        for (i, v) in va.iter().enumerate() {
            let val = (v as f64) * e10;
            append(&mut dst, i + ofs, val);
        }
    }
}