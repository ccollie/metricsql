use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// A query value type
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    /// A 64-bit floating point number.
    Scalar,
    /// An owned String
    String,
    #[default]
    InstantVector,
    RangeVector,
}

impl ValueType {
    /// Returns true if this `ValueType` is a valid sub-expression of an
    /// operator, false if not.
    pub const fn is_operator_valid(&self) -> bool {
        match self {
            ValueType::Scalar |
            ValueType::String |
            ValueType::RangeVector | // ???????
            ValueType::InstantVector => true,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            ValueType::Scalar => "Scalar",
            ValueType::String => "String",
            ValueType::InstantVector => "InstantVector",
            ValueType::RangeVector => "RangeVector",
        }
    }

    pub const fn is_scalar(&self) -> bool {
        matches!(self, ValueType::Scalar)
    }
}

impl Display for ValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub trait Value {
    fn value_type(&self) -> ValueType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type() {
        assert_eq!(ValueType::Scalar.to_string(), "Scalar");
        assert_eq!(ValueType::String.to_string(), "String");
        assert_eq!(ValueType::InstantVector.to_string(), "InstantVector");
        assert_eq!(ValueType::RangeVector.to_string(), "RangeVector");
    }
}
