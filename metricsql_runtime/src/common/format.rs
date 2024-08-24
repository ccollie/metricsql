use std::fmt;
use std::fmt::Formatter;

pub fn format_number(f: &mut Formatter<'_>, value: f64) -> fmt::Result {
    if value.is_finite() {
        write!(f, "{}", value)
    } else if value.is_nan() {
        write!(f, "NaN")
    } else if value.is_sign_positive() {
        write!(f, "+Inf")
    } else {
        write!(f, "-Inf")
    }
}
