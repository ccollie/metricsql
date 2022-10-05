use std::fmt::{Display, Formatter};
use std::str::FromStr;
use crate::ast::ReturnValue;

use crate::error::Error;
use crate::parser::ParseError;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum DataType {
    RangeVector,
    InstantVector,
    /// Vec<f64> (normally Timeseries::values)
    Vector,
    /// A 64-bit floating point number.
    Scalar,
    /// An owned String
    String
}

impl DataType {
    /// Returns true if this type is numeric
    pub fn is_numeric(&self) -> bool {
        *self == DataType::Scalar
    }

    /// Returns true if this `ReturnKind` is a valid sub-expression of an
    /// operator, false if not.
    pub fn is_operator_valid(&self) -> bool {
        match self {
            DataType::String |
            DataType::Scalar |
            DataType::InstantVector => true,
            _ => false
        }
    }

}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DataType::RangeVector => "RangeVector",  // range vector
            DataType::InstantVector => "InstantVector",  // instant vector
            DataType::Vector => "Vector",  // basically destructured instant vector
            DataType::String => "String",
            DataType::Scalar => "Scalar"
        };
        write!(f, "{}", name)
    }
}

impl TryFrom<ReturnValue> for DataType {
    type Error = ParseError;

    fn try_from(value: ReturnValue) -> Result<Self, Self::Error> {
        match value {
            ReturnValue::Unknown(_) => Err(ParseError::General(String::from("Unknown DataType"))),
            ReturnValue::Scalar => Ok(DataType::Scalar),
            ReturnValue::String => Ok(DataType::String),
            ReturnValue::InstantVector => Ok(DataType::InstantVector),
            ReturnValue::RangeVector => Ok(DataType::RangeVector)
        }
    }
}

impl FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rangevector" => Ok(DataType::RangeVector),
            "instantvector" => Ok(DataType::InstantVector),
            "vector" => Ok(DataType::Vector),
            "scalar" => Ok(DataType::Scalar),
            _ => Err(Error::new(format!("Invalid data type name: {}", s)))
        }
    }
}