use std::fmt::{Display, Formatter};

use crate::ast::Expression;

#[derive(Debug, Clone)]
pub struct UnknownCause {
    /// An explanation for why this Unknown was returned
    pub message: String,

    // would be nice to make this a ref, but we run into wrapping issues
    /// The expression that caused an Unknown to be returned
    pub expression: Expression
}

/// A predicted return datatype
#[derive(Debug, Clone)]
pub enum ReturnType {
    Unknown(Box<UnknownCause>),
    Scalar,
    String,
    InstantVector,
    RangeVector
}

impl ReturnType {
    pub fn unknown<S>(message: S, expression: Expression) -> Self
        where
            S: Into<String>
    {
        ReturnType::Unknown(Box::new(UnknownCause {
            message: message.into(),
            expression
        }))
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
            _ => false
        }
    }
}

impl Display for ReturnType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ReturnType::*;

        let s: String;
        let name = match self {
            Unknown(u) => {
                s = format!("<unknown> - {} : {:?}", u.message, u.expression);
                &s
            },
            Scalar => "Scalar",
            String => "String",
            InstantVector => "InstantVector",
            RangeVector => "RangeVector"
        };
        write!(f, "{}", name)?;
        Ok(())
    }
}
