use super::misc::write_list;
use crate::ast::{BExpression, Expression, ExpressionNode};
use crate::common::ReturnType;
use crate::utils::escape_ident;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};

/// WithExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>,
    pub expr: BExpression,
}

impl WithExpr {
    pub fn new(expr: impl ExpressionNode, was: Vec<WithArgExpr>) -> Self {
        let expression = Expression::cast(expr);
        WithExpr {
            expr: Box::new(expression),
            was,
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
    pub expr: Expression,
    pub is_function: bool,
}

impl WithArgExpr {
    pub fn new_function<S: Into<String>>(name: S, expr: Expression, args: Vec<String>) -> Self {
        WithArgExpr {
            name: name.into(),
            args,
            expr,
            is_function: true,
        }
    }

    pub fn new<S: Into<String>>(name: S, expr: Expression) -> Self {
        WithArgExpr {
            name: name.into(),
            args: vec![],
            expr,
            is_function: false,
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
