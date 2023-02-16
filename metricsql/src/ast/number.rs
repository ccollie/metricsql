use crate::ast::{format_num, Expression, ExpressionNode};
use crate::common::ReturnType;
use lib::hash_f64;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Neg;

// todo: number => scalar
/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct NumberExpr {
    /// value is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
    /// the original token value
    s: String,
}

impl NumberExpr {
    pub fn new(v: f64) -> Self {
        NumberExpr {
            value: v,
            s: format!("{}", v),
        }
    }
    pub fn return_type(&self) -> ReturnType {
        ReturnType::Scalar
    }
}

impl From<f64> for NumberExpr {
    fn from(value: f64) -> Self {
        NumberExpr::new(value)
    }
}

impl From<i64> for NumberExpr {
    fn from(value: i64) -> Self {
        NumberExpr::new(value as f64)
    }
}

impl From<usize> for NumberExpr {
    fn from(value: usize) -> Self {
        NumberExpr::new(value as f64)
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
}

impl Display for NumberExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.s.len() > 0 {
            write!(f, "{}", self.s)
        } else {
            format_num(f, self.value)
        }
    }
}

impl Hash for NumberExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}

impl Neg for NumberExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let value = -self.value;
        NumberExpr {
            value,
            s: value.to_string(),
        }
    }
}
