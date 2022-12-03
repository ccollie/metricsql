use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// A query value type
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    /// A 64-bit floating point number.
    Scalar,
    /// An owned String
    String,
    InstantVector,
    RangeVector,
}

impl ValueType {
    /// Returns true if this `ValueType` is a valid sub-expression of an
    /// operator, false if not.
    pub fn is_operator_valid(&self) -> bool {
        match self {
            ValueType::Scalar |
            ValueType::String |
            ValueType::RangeVector | // ???????
            ValueType::InstantVector => true,
        }
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            ValueType::Scalar => true,
            _ => false,
        }
    }
}

impl Default for ValueType {
    fn default() -> Self {
        ValueType::InstantVector
    }
}

impl Display for ValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ValueType::*;

        let name = match self {
            Scalar => "Scalar",
            String => "String",
            InstantVector => "InstantVector",
            RangeVector => "RangeVector",
        };
        write!(f, "{}", name)?;
        Ok(())
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