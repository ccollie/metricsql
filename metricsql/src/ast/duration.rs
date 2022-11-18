use std::fmt;
use std::fmt::{Display, Formatter};
use text_size::TextRange;
use crate::ast::{Expression, ExpressionNode, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;
use crate::lexer::duration_value;
use crate::parser::ParseError;

/// DurationExpr contains the duration
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct DurationExpr {
    pub s: String,
    pub span: TextRange,
    pub const_value: i64,
    pub requires_step: bool,
}

impl TryFrom<&str> for DurationExpr {
    type Error = ParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let last_ch: char = s.chars().rev().next().unwrap();
        let const_value = duration_value(&s, 1)?;
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';

        Ok(Self {
            s: s.to_string(),
            const_value,
            requires_step,
            span: TextRange::default(),
        })
    }
}

impl DurationExpr {
    pub fn new(s: &str, span: TextRange) -> DurationExpr {
        let last_ch: char = s.chars().rev().next().unwrap();
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';
        // todo: the following is icky
        let const_value = duration_value(s, 1).unwrap_or(0);

        DurationExpr {
            s: s.to_string(),
            const_value,
            requires_step,
            span,
        }
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn duration(&self, step: i64) -> i64 {
        if self.requires_step {
            self.const_value * step
        } else {
            self.const_value
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::Scalar
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.s)?;
        Ok(())
    }
}

impl ExpressionNode for DurationExpr {
    fn cast(self) -> Expression {
        Expression::Duration(self)
    }
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Duration
    }
}