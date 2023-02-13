use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::Deref;
use crate::ast::{Expression, ExpressionNode, ReturnType};
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};

/// StringExpr represents string expression.
#[derive(Debug, Clone, Default, Hash, PartialEq, Serialize, Deserialize)]
pub struct StringExpr {
    /// contains unquoted value for string expression.
    pub value: String,
    pub span: TextSpan,
}

impl StringExpr {
    pub fn new<S: Into<String>, TS: Into<TextSpan>>(s: S, span: TS) -> Self {
        StringExpr {
            value: s.into(),
            span: span.into()
        }
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    #[inline]
    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::String
    }
}

impl From<String> for StringExpr {
    fn from(s: String) -> Self {
        let range = TextSpan::new(0,s.len());
        StringExpr::new(s, range)
    }
}

impl From<&str> for StringExpr {
    fn from(s: &str) -> Self {
        let range = TextSpan::new(0,s.len());
        StringExpr::new(s, range)
    }
}

impl ExpressionNode for StringExpr {
    fn cast(self) -> Expression {
        Expression::String(self)
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", enquote::enquote('"', &*self.value))?;
        Ok(())
    }
}

impl Deref for StringExpr {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
