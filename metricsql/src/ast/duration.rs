use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{Expression, ExpressionNode, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;
use crate::lexer::{duration_value, TextSpan};
use crate::parser::ParseError;
use serde::{Serialize, Deserialize};

/// DurationExpr contains a duration
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurationExpr {
    pub text: String,
    pub span: TextSpan,
    pub value: i64,
    pub requires_step: bool,
}

impl TryFrom<&str> for DurationExpr {
    type Error = ParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let last_ch: char = s.chars().rev().next().unwrap();
        let const_value = duration_value(&s, 1)?;
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';

        Ok(Self {
            text: s.to_string(),
            value: const_value,
            requires_step,
            span: TextSpan::default(),
        })
    }
}

impl DurationExpr {
    pub fn new<S: Into<TextSpan>>(text: &str, span: S) -> DurationExpr {
        let last_ch: char = text.chars().rev().next().unwrap();
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';
        // todo: the following is icky
        let const_value = duration_value(text, 1).unwrap_or(0);

        DurationExpr {
            text: text.to_string(),
            value: const_value,
            requires_step,
            span: span.into(),
        }
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn duration(&self, step: i64) -> i64 {
        if self.requires_step {
            self.value * step
        } else {
            self.value
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::Scalar
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.text)?;
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