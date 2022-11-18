use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::Deref;
use text_size::{TextRange, TextSize};
use crate::ast::{Expression, ExpressionNode, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;

/// StringExpr represents string expression.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct StringExpr {
    /// S contains unquoted value for string expression.
    pub s: String,

    /// A composite string has non-empty tokens.
    /// They must be converted into S by expand_with_expr.
    // todo: SmallVec
    pub(crate) tokens: Option<Vec<String>>,
    pub span: TextRange
}

impl StringExpr {
    pub fn new<S: Into<String>>(s: S, span: TextRange) -> Self {
        StringExpr {
            s: s.into(),
            tokens: None,
            span
        }
    }

    pub fn from_tokens(tokens: Vec<String>, span: TextRange) -> Self {
        StringExpr {
            s: "".to_string(),
            tokens: Some(tokens),
            span
        }
    }

    pub fn len(&self) -> usize {
        self.s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    #[inline]
    pub fn value(&self) -> &str {
        &self.s
    }

    pub fn has_tokens(&self) -> bool {
        self.token_count() > 0
    }

    pub fn token_count(&self) -> usize {
        match &self.tokens {
            Some(v) => v.len(),
            None => 0,
        }
    }

    pub(crate) fn is_expanded(&self) -> bool {
        !self.s.is_empty() || self.token_count() > 0
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::String
    }
}

impl From<String> for StringExpr {
    fn from(s: String) -> Self {
        let range = TextRange::at(TextSize::from(0),
                                  TextSize::from(s.len() as u32));
        StringExpr::new(s, range)
    }
}

impl From<&str> for StringExpr {
    fn from(s: &str) -> Self {
        let range = TextRange::at(TextSize::from(0),
                                  TextSize::from(s.len() as u32));
        StringExpr::new(s, range)
    }
}

impl ExpressionNode for StringExpr {
    fn cast(self) -> Expression {
        Expression::String(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::String
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", enquote::enquote('"', &*self.s))?;
        Ok(())
    }
}

impl Deref for StringExpr {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.s
    }
}