use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnValue};
use crate::lexer::TextSpan;
use super::misc::write_list;
use crate::utils::escape_ident;
use serde::{Serialize, Deserialize};

/// WithExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>,
    pub expr: BExpression,
    pub span: TextSpan,
}

impl WithExpr {
    pub fn new<S: Into<TextSpan>>(expr: impl ExpressionNode, was: Vec<WithArgExpr>, span: S) -> Self {
        let expression = Expression::cast(expr);
        WithExpr {
            expr: Box::new(expression),
            was,
            span: span.into(),
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        self.expr.return_value()
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
}

/// withArgExpr represents a single entry from WITH expression.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
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

    pub fn return_value(&self) -> ReturnValue {
        self.expr.return_value()
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