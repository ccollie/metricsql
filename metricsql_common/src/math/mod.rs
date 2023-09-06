use std::cmp::Ordering;

pub fn is_inf(x: f64, sign: i8) -> bool {
    match sign.cmp(&0_i8) {
        Ordering::Greater => x == f64::INFINITY,
        Ordering::Less => x == f64::NEG_INFINITY,
        Ordering::Equal => x == f64::INFINITY || x == f64::NEG_INFINITY,
    }
}
