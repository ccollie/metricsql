use std::borrow::Cow;
use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, NumberExpr, ReturnValue};
use crate::ast::expression_kind::ExpressionKind;
use crate::ast::misc::{intersection, write_labels, write_list};
use crate::ast::operator::BinaryOp;
use crate::lexer::TextSpan;
use crate::parser::{ParseError, ParseResult};
use serde::{Serialize, Deserialize};

// See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
pub enum GroupModifierOp {
    On,
    Ignoring,
}

impl Display for GroupModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use GroupModifierOp::*;
        match self {
            On => write!(f, "on")?,
            Ignoring => write!(f, "ignoring")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for GroupModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use GroupModifierOp::*;

        match op.to_lowercase().as_str() {
            "on" => Ok(On),
            "ignoring" => Ok(Ignoring),
            _ => Err(ParseError::General(format!(
                "Unknown group_modifier op: {}",
                op
            ))),
        }
    }
}

/// An operator matching clause
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct GroupModifier {
    /// Action applied to a list of vectors; whether `on (…)` or `ignored(…)` is used after the operator.
    pub op: GroupModifierOp,

    /// A list of labels to which the operator is applied
    pub labels: Vec<String>,
}

impl GroupModifier {
    pub fn new(op: GroupModifierOp, labels: Vec<String>) -> Self {
        GroupModifier {
            op,
            labels,
        }
    }

    /// Creates a GroupModifier cause with the On operator
    pub fn on(labels: Vec<String>) -> Self {
        GroupModifier::new(GroupModifierOp::On, labels)
    }

    /// Creates a GroupModifier clause using the Ignoring operator
    pub fn ignoring(labels: Vec<String>) -> Self {
        GroupModifier::new(GroupModifierOp::Ignoring, labels)
    }

    /// Replaces this GroupModifier's operator
    pub fn op(mut self, op: GroupModifierOp) -> Self {
        self.op = op;
        self
    }

    /// Adds a label key to this GroupModifier
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Replaces this GroupModifier's labels with the given set
    pub fn set_labels(&mut self, labels: Vec<String>) -> &mut Self {
        self.labels = labels;
        self
    }

    /// Clears this GroupModifier's set of labels
    pub fn clear_labels(&mut self) -> &mut Self {
        self.labels.clear();
        self
    }
}

impl Display for GroupModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} (", self.op)?;
        if !self.labels.is_empty() {
            write_list(&self.labels, f, false)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash, Serialize, Deserialize)]
pub enum VectorMatchCardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany
}

impl Display for VectorMatchCardinality {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use VectorMatchCardinality::*;
        let str = match self {
            OneToOne => "one-to-one",
            OneToMany => "one-to-many",
            ManyToOne => "many-to-one",
            ManyToMany => "many-to-many",
        };
        write!(f, "{}", str)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash, Serialize, Deserialize)]
pub enum JoinModifierOp {
    GroupLeft,
    GroupRight,
}

impl Display for JoinModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use JoinModifierOp::*;
        match self {
            GroupLeft => write!(f, "group_left")?,
            GroupRight => write!(f, "group_right")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for JoinModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use JoinModifierOp::*;

        match op.to_lowercase().as_str() {
            "group_left" => Ok(GroupLeft),
            "group_right" => Ok(GroupRight),
            _ => {
                let msg = format!("Unknown join_modifier op: {}", op);
                Err(ParseError::General(msg))
            }
        }
    }
}

/// A JoinModifier clause's nested grouping clause
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct JoinModifier {
    /// The GroupModifier group's operator type (left or right)
    pub op: JoinModifierOp,

    /// A list of labels to copy to the opposite side of the group operator, i.e.
    /// group_left(foo) copies the label `foo` from the right hand side
    pub labels: Vec<String>,

    /// The cardinality of the two Vectors.
    pub cardinality: VectorMatchCardinality,
}

impl JoinModifier {
    pub fn new(op: JoinModifierOp, labels: Vec<String>) -> Self {
        JoinModifier {
            op,
            labels,
            cardinality: VectorMatchCardinality::OneToOne
        }
    }

    /// Creates a new JoinModifier with the Left op
    pub fn left() -> Self {
        JoinModifier::new(JoinModifierOp::GroupLeft, vec![])
    }

    /// Creates a new JoinModifier with the Right op
    pub fn right() -> Self {
        JoinModifier::new(JoinModifierOp::GroupRight, vec![])
    }

    /// Replaces this JoinModifier's operator
    pub fn op(mut self, op: JoinModifierOp) -> Self {
        self.op = op;
        self
    }

    /// Adds a label key to this JoinModifier
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Replaces this JoinModifier's labels with the given set
    pub fn set_labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|l| (*l).to_string()).collect();
        self
    }

    /// Clears this JoinModifier's set of labels
    pub fn clear_labels(mut self) -> Self {
        self.labels.clear();
        self
    }
}

impl Display for JoinModifier {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} (", self.op)?;
        if !self.labels.is_empty() {
            write_labels(&self.labels, f)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct BinaryOpExpr {
    /// Op is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: BinaryOp,

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

    pub span: TextSpan,
}

impl BinaryOpExpr {
    pub fn new(op: BinaryOp, lhs: Expression, rhs: Expression) -> ParseResult<Self> {
        let expr = BinaryOpExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            join_modifier: None,
            group_modifier: None,
            bool_modifier: false,
            span: TextSpan::default(),
        };

        // ensure we have a operands are valid for the operator
        match expr.return_value() {
            ReturnValue::Unknown(unknown_cause) => {
                // todo: better error variant. also include span
                Err(ParseError::General(
                    unknown_cause.message
                ))
            },
            _ => Ok(expr)
        }
    }

    /// Unary minus. Substitute `-expr` with `0 - expr`
    pub fn new_unary_minus<S: Into<TextSpan>>(e: impl ExpressionNode, span: S) -> ParseResult<Self> {
        let expr = Expression::cast(e);
        let lhs = Expression::Number(NumberExpr::new(0.0, span));
        BinaryOpExpr::new(BinaryOp::Sub, lhs, expr)
    }

    pub fn get_group_modifier_or_default(&self) -> (GroupModifierOp, Cow<Vec<String>>) {
        match &self.group_modifier {
            None => {
                (GroupModifierOp::Ignoring, Cow::Owned::<Vec<String>>(vec![]))
            },
            Some(modifier) => {
                (modifier.op, Cow::Borrowed(&modifier.labels))
            }
        }
    }

    /// Convert 'num cmpOp query' expression to `query reverseCmpOp num` expression
    /// like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
    /// in order to return valid values for `foo` that are bigger than 0.5.
    pub fn adjust_comparison_op(&mut self) -> bool {
        if self.should_adjust_comparison_op() {
            self.op = self.op.get_reverse_cmp();
            std::mem::swap(&mut self.left, &mut self.right);
            return true;
        }
        return false
    }

    pub fn should_adjust_comparison_op(&self) -> bool {
        if !self.op.is_comparison() {
            return false
        }

        if Expression::is_number(&self.right) || !Expression::is_scalar(&self.left) {
            return false
        }
        true
    }

    pub fn return_value(&self) -> ReturnValue {
        // binary operator exprs can only contain (and return) instant vectors
        let lhs_ret = self.left.return_value();
        let rhs_ret = self.right.return_value();

        // operators can only have instant vectors or scalars
        if !lhs_ret.is_operator_valid() {
            return ReturnValue::unknown(
                format!("lhs return type ({:?}) is not valid in an operator", &lhs_ret),
                self.clone().cast()
            );
        }

        if !rhs_ret.is_operator_valid() {
            return ReturnValue::unknown(
                format!("rhs return type ({:?}) is not valid in an operator", &rhs_ret),
                self.clone().cast()
            );
        }

        match (lhs_ret, rhs_ret) {
            (ReturnValue::Scalar, ReturnValue::Scalar) => ReturnValue::Scalar,
            (ReturnValue::RangeVector, ReturnValue::RangeVector) => ReturnValue::RangeVector,
            (ReturnValue::String, ReturnValue::String) => {
                if self.op != BinaryOp::Add {
                    return ReturnValue::unknown(
                        format!("Operator {} is not valid for (String, String)", self.op),
                        self.clone().cast()
                    );
                }
                return ReturnValue::String
            },
            _ => return ReturnValue::InstantVector
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
                        let msg = format!("labels ({}) must not occur in ON and GROUP clause at once", duplicates.join(", "));
                        return Err(ParseError::General(msg))
                    }
                }
            },
            _ => {
            }
        }
        Ok(())
    }
}

impl Display for BinaryOpExpr {
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

impl ExpressionNode for BinaryOpExpr {
    fn cast(self) -> Expression {
        Expression::BinaryOperator(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Binop
    }
}

// Gets vector matching cardinality. Assumes both operands are InstantVector
pub fn get_vector_match_cardinality(binary_node: &BinaryOpExpr) -> VectorMatchCardinality {
    let mut card = VectorMatchCardinality::OneToOne;
    if let Some(join_modifier) = &binary_node.join_modifier {
        let group_left = join_modifier.op == JoinModifierOp::GroupLeft;
        if group_left {
            card = VectorMatchCardinality::ManyToOne
        } else {
            card = VectorMatchCardinality::OneToMany
        }
    };
    if binary_node.op.is_set_operator() && card == VectorMatchCardinality::OneToOne {
        card = VectorMatchCardinality::ManyToMany;
    }
    return card
}
