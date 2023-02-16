use crate::ast::{Expression, ExpressionNode, SegmentedString, StringSegment};
use crate::common::ReturnType;
use crate::parser::ParseResult;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};

/// StringExpr represents string expression.
#[derive(Debug, Clone, Default, Hash, PartialEq, Serialize, Deserialize)]
pub struct StringExpr(SegmentedString);

impl StringExpr {
    pub fn new<S: Into<String>>(s: S) -> Self {
        StringExpr(SegmentedString::from(s.into()))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn push_str(&mut self, value: &str) {
        self.0.push_str(value);
    }

    pub fn push_ident(&mut self, value: &str) {
        self.0.push_ident(value);
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::String
    }

    pub fn is_resolved(&self) -> bool {
        self.0.is_expanded()
    }

    pub fn is_literal_only(&self) -> bool {
        self.0.is_literal_only()
    }

    pub fn resolve<F>(&self, f: F) -> ParseResult<String>
    where
        F: Fn(&str) -> ParseResult<&str>,
    {
        self.0.resolve(f)
    }

    pub fn estimate_result_capacity(&self) -> usize {
        self.0.estimate_result_capacity()
    }

    pub fn iter(&self) -> impl Iterator<Item = &StringSegment> + '_ {
        self.0.iter()
    }
}

impl From<String> for StringExpr {
    fn from(s: String) -> Self {
        StringExpr::new(s)
    }
}

impl From<&str> for StringExpr {
    fn from(s: &str) -> Self {
        StringExpr::new(s)
    }
}

impl ExpressionNode for StringExpr {
    fn cast(self) -> Expression {
        Expression::String(self)
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)?;
        Ok(())
    }
}
