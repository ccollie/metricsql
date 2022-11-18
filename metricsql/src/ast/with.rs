use std::fmt;
use std::fmt::{Display, Formatter};
use text_size::{TextRange};
use crate::ast::{BExpression, Expression, ExpressionNode};
use crate::ast::expression_kind::ExpressionKind;
use super::misc::write_list;
use crate::utils::escape_ident;

/// WithExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, Hash)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>,
    pub expr: BExpression,
    pub span: TextRange,
}

impl WithExpr {
    pub fn new(expr: impl ExpressionNode, was: Vec<WithArgExpr>, span: TextRange) -> Self {
        let expression = Expression::cast(expr);
        WithExpr {
            expr: Box::new(expression),
            was,
            span,
        }
    }
}

impl Display for WithExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "WITH (")?;
        for (i, was) in self.was.iter().enumerate() {
            if (i + 1) < self.was.len() {
                write!(f, ", ")?;
            }
            write!(f, "{}", was)?;
        }
        write!(f, ") ")?;
        write!(f, "{}", self.expr)?;
        Ok(())
    }
}

impl ExpressionNode for WithExpr {
    fn cast(self) -> Expression {
        Expression::With(self)
    }
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::With
    }
}

/// withArgExpr represents a single entry from WITH expression.
#[derive(Debug, Clone, Hash)]
pub struct WithArgExpr {
    pub name: String,
    pub args: Vec<String>,
    pub expr: BExpression,
}

impl WithArgExpr {
    pub fn new<S: Into<String>>(name: S, expr: Expression, args: Vec<String>) -> Self {
        WithArgExpr {
            name: name.into(),
            args,
            expr: Box::new(expr),
        }
    }
}

impl Display for WithArgExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.name))?;
        write_list(&self.args, f, !self.args.is_empty())?;
        write!(f, " = ")?;
        write!(f, "{}", self.expr)?;
        Ok(())
    }
}