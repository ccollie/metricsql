use crate::ast::misc::write_expression_list;
use crate::ast::{BExpression, Expression, ExpressionNode};
use crate::common::ReturnType;
use crate::functions::TransformFunction;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Display, Formatter};

// TODO: ParensExpr => GroupExpr
/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct ParensExpr {
    pub expressions: Vec<BExpression>,
}

impl ParensExpr {
    pub fn new(expressions: Vec<BExpression>) -> Self {
        ParensExpr { expressions }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }

    pub fn return_type(&self) -> ReturnType {
        if self.expressions.len() == 1 {
            return self.expressions[0].return_type();
        }

        // Treat as a function with empty name, i.e. union()
        TransformFunction::Union.return_type()
    }
}

impl Display for ParensExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_expression_list(&self.expressions, f)?;
        Ok(())
    }
}

impl ExpressionNode for ParensExpr {
    fn cast(self) -> Expression {
        Expression::Parens(self)
    }
}
