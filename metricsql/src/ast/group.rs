use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnValue};
use crate::ast::misc::write_expression_list;
use crate::functions::TransformFunction;
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};

// TODO: ParensExpr => GroupExpr
/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, Hash, Serialize, Deserialize)]
pub struct ParensExpr {
    pub expressions: Vec<BExpression>,
    pub span: TextSpan,
}

impl ParensExpr {
    pub fn new<S: Into<TextSpan>>(expressions: Vec<BExpression>, span: S) -> Self {
        ParensExpr {
            expressions,
            span: span.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }

    pub fn return_value(&self) -> ReturnValue {
        if self.expressions.len() == 1 {
            return self.expressions[0].return_value();
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