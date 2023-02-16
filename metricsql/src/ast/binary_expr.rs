use crate::ast::misc::intersection;
use crate::ast::{BExpression, Expression, ExpressionNode};
use crate::common::{
    GroupModifier, GroupModifierOp, JoinModifier, JoinModifierOp, Operator, ReturnType,
    VectorMatchCardinality,
};
use crate::parser::{ParseError, ParseResult};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;
use std::fmt::{Display, Formatter};

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct BinaryExpr {
    /// Op is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: Operator,

    /// bool_modifier indicates whether `bool` modifier is present.
    /// For example, `foo > bool bar`.
    pub bool_modifier: bool,

    /// group_modifier contains modifier such as "on" or "ignoring".
    pub group_modifier: Option<GroupModifier>,

    /// join_modifier contains modifier such as "group_left" or "group_right".
    pub join_modifier: Option<JoinModifier>,

    /// left contains left arg for the `left op right` expression.
    pub left: BExpression,

    /// right contains right arg for the `left op right` expression.
    pub right: BExpression,
}

impl BinaryExpr {
    pub fn new(op: Operator, lhs: Expression, rhs: Expression) -> Self {
        BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            join_modifier: None,
            group_modifier: None,
            bool_modifier: false,
        }
    }

    /// Unary minus. Substitute `-expr` with `0 - expr`
    pub fn new_unary_minus(e: impl ExpressionNode) -> Self {
        let expr = Expression::cast(e);
        let lhs = Expression::from(0.0);
        BinaryExpr::new(Operator::Sub, lhs, expr)
    }

    pub fn get_group_modifier_or_default(&self) -> (GroupModifierOp, Cow<Vec<String>>) {
        match &self.group_modifier {
            None => (GroupModifierOp::Ignoring, Cow::Owned::<Vec<String>>(vec![])),
            Some(modifier) => (modifier.op, Cow::Borrowed(&modifier.labels)),
        }
    }

    pub fn with_bool_modifier(mut self) -> Self {
        self.bool_modifier = true;
        self
    }

    /// Convert `num cmpOp query` expression to `query reverseCmpOp num` expression
    /// like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
    /// in order to return valid values for `foo` that are bigger than 0.5.
    pub fn adjust_comparison_op(&mut self) -> bool {
        if self.should_adjust_comparison_op() {
            self.op = self.op.get_reverse_cmp();
            std::mem::swap(&mut self.left, &mut self.right);
            return true;
        }
        return false;
    }

    pub fn should_adjust_comparison_op(&self) -> bool {
        if !self.op.is_comparison() {
            return false;
        }

        if Expression::is_number(&self.right) || !Expression::is_scalar(&self.left) {
            return false;
        }
        true
    }

    pub fn return_type(&self) -> ReturnType {
        // binary operator expressions can only contain (and return) instant vectors
        let lhs_ret = self.left.return_type();
        let rhs_ret = self.right.return_type();

        // operators can only have instant vectors or scalars
        if !lhs_ret.is_operator_valid() {
            return ReturnType::unknown(
                format!(
                    "lhs return type ({:?}) is not valid in an operator",
                    &lhs_ret
                ),
                self.to_string(),
            );
        }

        if !rhs_ret.is_operator_valid() {
            return ReturnType::unknown(
                format!(
                    "rhs return type ({:?}) is not valid in an operator",
                    &rhs_ret
                ),
                self.to_string(),
            );
        }

        match (lhs_ret, rhs_ret) {
            (ReturnType::Scalar, ReturnType::Scalar) => ReturnType::Scalar,
            (ReturnType::RangeVector, ReturnType::RangeVector) => ReturnType::RangeVector,
            (ReturnType::String, ReturnType::String) => {
                if self.op.is_comparison() {
                    return ReturnType::Scalar;
                }
                if self.op != Operator::Add {
                    return ReturnType::unknown(
                        format!("Operator {} is not valid for (String, String)", self.op),
                        self.to_string(),
                    );
                }
                return ReturnType::String;
            }
            _ => return ReturnType::InstantVector,
        }
    }

    pub fn vector_match_cardinality(&self) -> Option<VectorMatchCardinality> {
        if self.join_modifier.is_some() {
            // todo: ensure both left and right return instant vectors
            return Some(get_vector_match_cardinality(&self));
        }
        None
    }

    pub fn validate_modifier_labels(&self) -> ParseResult<()> {
        match (&self.group_modifier, &self.join_modifier) {
            (Some(group_modifier), Some(join_modifier)) => {
                if group_modifier.op == GroupModifierOp::On {
                    let duplicates = intersection(&group_modifier.labels, &join_modifier.labels);
                    if !duplicates.is_empty() {
                        let msg = format!(
                            "labels ({}) must not occur in ON and GROUP clause at once",
                            duplicates.join(", ")
                        );
                        return Err(ParseError::General(msg));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Display for BinaryExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
        if self.left.is_binary_op() {
            write!(f, "({})", self.left)?;
        } else {
            write!(f, "{}", self.left)?;
        }
        write!(f, " {}", self.op)?;
        if self.bool_modifier {
            write!(f, " bool")?;
        }
        if let Some(modifier) = &self.group_modifier {
            write!(f, " {}", modifier)?;
        }
        if let Some(modifier) = &self.join_modifier {
            write!(f, " {}", modifier)?;
        }
        write!(f, " ")?;
        if self.right.is_binary_op() {
            write!(f, "({})", self.right)?;
        } else {
            write!(f, "{}", self.right)?;
        }
        Ok(())
    }
}

impl ExpressionNode for BinaryExpr {
    fn cast(self) -> Expression {
        Expression::BinaryOperator(self)
    }
}

/// Gets vector matching cardinality. Assumes both operands are InstantVector
pub fn get_vector_match_cardinality(binary_node: &BinaryExpr) -> VectorMatchCardinality {
    let mut card = VectorMatchCardinality::OneToOne;
    if let Some(join_modifier) = &binary_node.join_modifier {
        if join_modifier.op == JoinModifierOp::GroupLeft {
            card = VectorMatchCardinality::ManyToOne
        } else {
            card = VectorMatchCardinality::OneToMany
        }
    };
    if binary_node.op.is_set_operator() && card == VectorMatchCardinality::OneToOne {
        card = VectorMatchCardinality::ManyToMany;
    }
    return card;
}
