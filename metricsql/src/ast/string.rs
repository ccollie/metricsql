use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::Deref;
use crate::ast::{Expression, ExpressionNode, ReturnValue};
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Hash)]
pub enum StringTokenType {
    String(String),
    Ident(String)
}

/// StringExpr represents string expression.
#[derive(Debug, Clone, Default, Hash, Serialize, Deserialize)]
pub struct StringExpr {
    /// contains unquoted value for string expression.
    pub value: String,
    pub span: TextSpan,

    /// A composite string has non-empty tokens.
    /// They must be converted into S by expand_with_expr.
    // todo: SmallVec
    #[serde(skip)]
    pub(crate) tokens: Vec<StringTokenType>,
}

impl StringExpr {
    pub fn new<S: Into<String>, TS: Into<TextSpan>>(s: S, span: TS) -> Self {
        StringExpr {
            value: s.into(),
            tokens: vec![],
            span: span.into()
        }
    }

    pub fn from_string<TS: Into<TextSpan>>(tokens: Vec<String>, span: TS) -> Self {
        let toks = tokens.iter()
            .map(|x| StringTokenType::String(x.into()))
            .collect::<Vec<StringTokenType>>();

        StringExpr {
            value: "".to_string(),
            tokens: toks,
            span: span.into()
        }
    }

    pub fn from_tokens<TS: Into<TextSpan>>(tokens: Vec<StringTokenType>, span: TS) -> Self {
        StringExpr {
            value: "".to_string(),
            tokens,
            span: span.into()
        }
    }

    pub fn add_string(&mut self, tok: &str) {
        self.tokens.push(StringTokenType::String(tok.to_string()))
    }

    pub fn add_ident(&mut self, tok: &str) {
        self.tokens.push(StringTokenType::Ident(tok.to_string()))
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

    pub fn has_tokens(&self) -> bool {
        self.token_count() > 0
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn is_expanded(&self) -> bool {
        !self.value.is_empty() || self.token_count() == 0
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::String
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
