use std::hash::Hasher;

/// Returns the mantissa, exponent and sign as integers.
pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = val.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

pub fn hash_f64<H: Hasher>(state: &mut H, value: f64) {
    let (mantissa, exponent, sign) = integer_decode(value);
    state.write_u64(mantissa);
    state.write_i16(exponent);
    state.write_i8(sign)
}