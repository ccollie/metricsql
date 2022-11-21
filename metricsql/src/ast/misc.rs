use std::{fmt};
use std::collections::HashSet;
use std::fmt::Formatter;
use crate::ast::{BExpression};
use crate::utils::escape_ident;


pub(super) fn write_expression_list(exprs: &[BExpression], f: &mut Formatter) -> Result<(), fmt::Error> {
    let mut items: Vec<String> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        items.push(format!("{}", expr));
    }
    write_list(&items, f, true)?;
    Ok(())
}

pub(super) fn write_list(
    values: &Vec<String>,
    f: &mut Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.iter().enumerate() {
        if (i + 1) < values.len() {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    if use_parens {
        write!(f, ")")?;
    }
    Ok(())
}

pub(super) fn write_labels(labels: &[String], f: &mut Formatter) -> Result<(), fmt::Error> {
    if !labels.is_empty() {
        write!(f, "(")?;
        for (i, label) in labels.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", escape_ident(label))?;
        }
        write!(f, ")")?;
    }
    Ok(())
}

pub fn intersection(labels_a: &Vec<String>, labels_b: &Vec<String>) -> Vec<String> {
    if labels_a.is_empty() || labels_b.is_empty() {
        return vec![]
    }
    let unique_a: HashSet<String> = labels_a.clone().into_iter().collect();
    let unique_b: HashSet<String> = labels_b.clone().into_iter().collect();
    unique_a
        .intersection(&unique_b)
        .map(|i| i.clone())
        .collect::<Vec<_>>()
}
