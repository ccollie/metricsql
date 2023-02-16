use crate::ast::BExpression;
use crate::utils::escape_ident;
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Display, Formatter};

pub(crate) fn write_expression_list(
    exprs: &[BExpression],
    f: &mut Formatter,
) -> Result<(), fmt::Error> {
    let mut items: Vec<String> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        items.push(format!("{}", expr));
    }
    write_list(&items, f, true)?;
    Ok(())
}

pub(crate) fn write_list<T: Display>(
    values: &Vec<T>,
    f: &mut Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.iter().enumerate() {
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

pub(crate) fn write_labels(labels: &[String], f: &mut Formatter) -> Result<(), fmt::Error> {
    write!(f, "(")?;
    for (i, label) in labels.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", escape_ident(label))?;
    }
    write!(f, ")")?;
    Ok(())
}

pub fn intersection(labels_a: &Vec<String>, labels_b: &Vec<String>) -> Vec<String> {
    if labels_a.is_empty() || labels_b.is_empty() {
        return vec![];
    }
    let unique_a: HashSet<String> = labels_a.clone().into_iter().collect();
    let unique_b: HashSet<String> = labels_b.clone().into_iter().collect();
    unique_a
        .intersection(&unique_b)
        .map(|i| i.clone())
        .collect::<Vec<_>>()
}

pub(crate) fn format_num(f: &mut Formatter<'_>, value: f64) -> fmt::Result {
    if value.is_nan() {
        write!(f, "NaN")
    } else if value.is_finite() {
        write!(f, "{}", value)
    } else if value.is_sign_positive() {
        write!(f, "+Inf")
    } else {
        write!(f, "-Inf")
    }
}
