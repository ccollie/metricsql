use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::error::Error;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum DataType {
    Matrix,
    Series,
    /// Vec<f64> (normally Timeseries::values)
    Vector,
    /// A 64-bit floating point number.
    Float,
    /// A 64-bit int.
    Int,
    /// An owned String
    String
}

impl DataType {
    /// Returns true if this type is numeric: (Int or Float).
    pub fn is_numeric(&self) -> bool {
        *self == DataType::Float || *self == DataType::Int
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DataType::Matrix => "Matrix",  // range vector
            DataType::Series => "Series",  // instant vector
            DataType::Vector => "Vector",  // basically destructured instant vector
            DataType::Float => "Float",
            DataType::Int => "Int",
            DataType::String => "String"
        };
        write!(f, "{}", name)
    }
}

impl FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "matrix" => Ok(DataType::Matrix),
            "series" => Ok(DataType::Series),
            "vector" => Ok(DataType::Vector),
            "float" => Ok(DataType::Float),
            "int" => Ok(DataType::Int),
            _ => Err(Error::new(format!("Invalid data type name: {}", s)))
        }
    }
}