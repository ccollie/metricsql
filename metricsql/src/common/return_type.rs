use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct UnknownCause {
    /// An explanation for why this Unknown was returned
    pub message: String,

    // would be nice to make this a ref, but we run into wrapping issues
    pub expression: String,
}

/// A predicted return datatype
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub enum ReturnType {
    Unknown(UnknownCause),
    Scalar,
    String,
    InstantVector,
    RangeVector,
}

impl ReturnType {
    pub fn unknown<S>(message: S, expression: String) -> Self
    where
        S: Into<String>,
    {
        ReturnType::Unknown(UnknownCause {
            message: message.into(),
            expression,
        })
    }

    /// Returns true if this `ReturnValue` is a valid sub-expression of an
    /// operator, false if not.
    pub fn is_operator_valid(&self) -> bool {
        match self {
            ReturnType::Scalar |
            ReturnType::String |
            ReturnType::RangeVector | // ???????
            ReturnType::InstantVector => true,
            _ => false
        }
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            ReturnType::Scalar => true,
            _ => false,
        }
    }
}

impl Default for ReturnType {
    fn default() -> Self {
        ReturnType::InstantVector
    }
}

impl Display for ReturnType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ReturnType::*;

        let name = match self {
            Unknown(u) => {
                write!(f, "<unknown> - {} : {}", u.message, u.expression)?;
                return Ok(());
            }
            Scalar => "Scalar",
            String => "String",
            InstantVector => "InstantVector",
            RangeVector => "RangeVector",
        };
        write!(f, "{}", name)?;
        Ok(())
    }
}
