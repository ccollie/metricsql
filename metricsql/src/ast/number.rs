use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use text_size::TextRange;
use lib::hash_f64;
use crate::ast::{Expression, ExpressionNode, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;

// todo: number => scalar
/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Copy)]
pub struct NumberExpr {
    /// value is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
    pub span: TextRange
}

impl NumberExpr {
    pub fn new(v: f64, span: TextRange) -> Self {
        NumberExpr { value: v, span }
    }
    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::Scalar
    }
}

impl From<f64> for NumberExpr {
    fn from(value: f64) -> Self {
        NumberExpr::new(value, TextRange::default())
    }
}

impl From<i64> for NumberExpr {
    fn from(value: i64) -> Self {
        NumberExpr::new(value as f64, TextRange::default())
    }
}

impl From<usize> for NumberExpr {
    fn from(value: usize) -> Self {
        NumberExpr::new(value as f64, TextRange::default())
    }
}

impl Into<f64> for NumberExpr {
    fn into(self) -> f64 {
        self.value
    }
}

impl PartialEq for NumberExpr {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() <= f64::EPSILON
    }
}

impl ExpressionNode for NumberExpr {
    fn cast(self) -> Expression {
        Expression::Number(self)
    }
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Number
    }
}

impl Display for NumberExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.value.is_nan() {
            write!(f, "NaN")?;
        } else if self.value.is_finite() {
            write!(f, "{}", self.value)?;
        } else if self.value.is_sign_positive() {
            write!(f, "+Inf")?;
        } else {
            write!(f, "-Inf")?;
        }
        Ok(())
    }
}

impl Hash for NumberExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}