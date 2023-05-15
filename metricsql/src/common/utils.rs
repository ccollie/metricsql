use std::fmt;
use std::fmt::{Display, Formatter};

pub(crate) fn format_num(f: &mut Formatter<'_>, value: f64) -> fmt::Result {
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

pub(crate) fn write_list<T: Display>(
    values: impl Iterator<Item = T>,
    f: &mut Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    if use_parens {
        write!(f, ")")?;
    }
    Ok(())
}
