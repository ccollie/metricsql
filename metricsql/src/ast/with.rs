use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnType};
use crate::lexer::TextSpan;
use super::misc::write_list;
use crate::utils::escape_ident;
use serde::{Serialize, Deserialize};

/// WithExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
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

    pub fn return_type(&self) -> ReturnType {
        self.expr.return_type()
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

/// WithArgExpr represents a single entry from WITH expression.
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct WithArgExpr {
    pub name: String,
    pub args: Vec<String>,
    pub expr: BExpression,
    pub is_function: bool,
}

impl WithArgExpr {
    pub fn new_function<S: Into<String>>(name: S, expr: Expression, args: Vec<String>) -> Self {
        WithArgExpr {
            name: name.into(),
            args,
            expr: Box::new(expr),
            is_function: true
        }
    }

    pub fn new<S: Into<String>>(name: S, expr: Expression) -> Self {
        WithArgExpr {
            name: name.into(),
            args: vec![],
            expr: Box::new(expr),
            is_function: false
        }
    }

    pub fn return_value(&self) -> ReturnType {
        self.expr.return_type()
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

pub(crate) enum WithExprParam {
    Function(WithExpr),
    Value(WithArgExpr)
}

impl WithExprParam {
    pub fn function(expr: WithExpr) -> Self {
        WithExprParam::Function(expr)
    }
    
    pub fn value(arg: WithArgExpr) -> Self {
        WithExprParam::Value(arg)
    }

    pub fn expr(&self) -> &Expression {
        match self {
            WithExprParam::Function(func) => &func.expr,
            WithExprParam::Value(val) => &val.expr
        }
    }
}