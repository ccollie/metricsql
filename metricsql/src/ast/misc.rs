use std::fmt;
use std::fmt::Formatter;
use text_size::TextRange;
use crate::ast::{BExpression, DurationExpr};
use crate::utils::escape_ident;

/// A Subquery which converts an instant vector to a range vector by repeatedly
/// evaluating it at set intervals into the relative past
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Subquery {
    /// Duration back in time to begin the subquery
    pub range: DurationExpr,

    /// Optional step size. If unset, uses the global/query default at runtime.
    pub resolution: Option<DurationExpr>,

    pub span: Option<TextRange>
}

impl Subquery {
    pub fn new(range: DurationExpr) -> Self {
        Subquery {
            range,
            resolution: None,
            span: None
        }
    }

    pub fn resolution(mut self, res: DurationExpr) -> Self {
        self.resolution = Some(res);
        self
    }

    pub fn clear_resolution(mut self) -> Self {
        self.resolution = None;
        self
    }

    pub fn span<S: Into<TextRange>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl fmt::Display for Subquery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(res) = &self.resolution {
            write!(f, "[{}:{}]", self.range, res)
        } else {
            write!(f, "[{}:]", self.range)
        }
    }
}


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

