use crate::ast::{Expression, ExpressionNode};
use crate::common::ReturnType;
use crate::lexer::parse_duration_value;
use crate::parser::{ParseError, ParseResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};

/// DurationExpr contains a duration
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurationExpr {
    pub text: String,
    pub value: i64,
    pub requires_step: bool,
}

impl TryFrom<&str> for DurationExpr {
    type Error = ParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let last_ch: char = s.chars().last().unwrap();
        let const_value = parse_duration_value(&s, 1)?;
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';

        Ok(Self {
            text: s.to_string(),
            value: const_value,
            requires_step,
        })
    }
}

impl DurationExpr {
    pub fn new(text: &str) -> ParseResult<DurationExpr> {
        let last_ch: char = text.chars().last().unwrap();
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';
        // todo: the following is icky
        let const_value = parse_duration_value(text, 1)?;

        Ok(DurationExpr {
            text: text.to_string(),
            value: const_value,
            requires_step,
        })
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn value(&self, step: i64) -> i64 {
        if self.requires_step {
            self.value * step
        } else {
            self.value
        }
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::Scalar
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
}
