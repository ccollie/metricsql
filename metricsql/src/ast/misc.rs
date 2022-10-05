use std::fmt;
use text_size::TextRange;

use crate::ast::{DurationExpr};

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