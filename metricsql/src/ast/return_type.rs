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
pub enum ReturnValue {
    Unknown(Box<UnknownCause>),
    Scalar,
    String,
    InstantVector,
    RangeVector
}

impl ReturnValue {
    pub fn unknown<S>(message: S, expression: Expression) -> Self
        where
            S: Into<String>
    {
        ReturnValue::Unknown(Box::new(UnknownCause {
            message: message.into(),
            expression
        }))
    }

    /// Returns true if this `ReturnValue` is a valid sub-expression of an
    /// operator, false if not.
    pub fn is_operator_valid(&self) -> bool {
        match self {
            ReturnValue::Scalar | ReturnValue::String | ReturnValue::InstantVector => true,
            _ => false
        }
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            ReturnValue::Scalar => true,
            _ => false
        }
    }
}

impl Display for ReturnValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s: String;
        let name = match self {
            ReturnValue::Unknown(u) => {
                s = format!("<unknown> - {} : {:?}", u.message, u.expression);
                &s
            },
            ReturnValue::Scalar => "Scalar",
            ReturnValue::String => "String",
            ReturnValue::InstantVector => "InstantVector",
            ReturnValue::RangeVector => "RangeVector"
        };
        write!(f, "{}", name)?;
        Ok(())
    }
}
