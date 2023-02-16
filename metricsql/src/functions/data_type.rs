use crate::common::ReturnType;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::parser::ParseError;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum DataType {
    RangeVector,
    InstantVector,
    /// A 64-bit floating point number.
    Scalar,
    /// An owned String
    String,
}

impl DataType {
    /// Returns true if this type is numeric
    pub fn is_numeric(&self) -> bool {
        *self == DataType::Scalar
    }

    /// Returns true if this `DataType` is a valid sub-expression of an
    /// operator, false if not.
    pub fn is_operator_valid(&self) -> bool {
        matches!(
            self,
            DataType::String | DataType::Scalar | DataType::InstantVector
        )
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DataType::RangeVector => "RangeVector",     // range vector
            DataType::InstantVector => "InstantVector", // instant vector
            DataType::String => "String",
            DataType::Scalar => "Scalar",
        };
        write!(f, "{}", name)
    }
}

impl TryFrom<ReturnType> for DataType {
    type Error = ParseError;

    fn try_from(value: ReturnType) -> Result<Self, Self::Error> {
        match value {
            ReturnType::Unknown(_) => Err(ParseError::General(String::from("Unknown DataType"))),
            ReturnType::Scalar => Ok(DataType::Scalar),
            ReturnType::String => Ok(DataType::String),
            ReturnType::InstantVector => Ok(DataType::InstantVector),
            ReturnType::RangeVector => Ok(DataType::RangeVector),
        }
    }
}

impl FromStr for DataType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rangevector" => Ok(DataType::RangeVector),
            "instantvector" => Ok(DataType::InstantVector),
            "string" => Ok(DataType::String),
            "scalar" => Ok(DataType::Scalar),
            _ => Err(ParseError::General(format!(
                "Invalid data type name: {}",
                s
            ))),
        }
    }
}
