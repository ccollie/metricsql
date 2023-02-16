use crate::ast::utils::visit_all;
use crate::ast::{
    AggrFuncExpr, BinaryExpr, DurationExpr, FuncExpr, MetricExpr, NumberExpr, ParensExpr,
    RollupExpr, StringExpr, WithExpr,
};
use crate::common::{Operator, ReturnType};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::string::String;
use std::{fmt, ops};

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn cast(self) -> Expression;
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    Duration(DurationExpr),

    /// A single scalar number.
    Number(NumberExpr),

    /// A single scalar string.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    String(StringExpr),

    /// A grouped expression wrapped in parentheses
    Parens(ParensExpr),

    /// A function call
    Function(FuncExpr),

    /// Aggregation represents aggregate functions such as `sum(...) by (...)`
    Aggregation(AggrFuncExpr),

    /// A binary operator expression
    BinaryOperator(BinaryExpr),

    /// A MetricsQL specific WITH statement node. Transformed at parse time to one
    /// of the other variants
    With(WithExpr),

    /// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
    Rollup(RollupExpr),

    /// MetricExpr represents a MetricsQL metric with optional filters, i.e. `foo{...}`.
    MetricExpression(MetricExpr),
}

pub type BExpression = Box<Expression>;

impl Expression {
    pub fn is_scalar(expr: &Expression) -> bool {
        match expr {
            Expression::Duration(_) | Expression::Number(_) => true,
            Expression::Function(f) => f.is_scalar,
            _ => false,
        }
    }

    pub fn cast(node: impl ExpressionNode) -> Self {
        node.cast()
    }

    pub fn is_number(expr: &Expression) -> bool {
        matches!(expr, Expression::Number(_))
    }

    pub fn contains_subquery(&self) -> bool {
        use Expression::*;
        match self {
            Function(fe) => fe.args.iter().any(|e| e.contains_subquery()),
            BinaryOperator(bo) => bo.left.contains_subquery() || bo.right.contains_subquery(),
            Aggregation(aggr) => aggr.args.iter().any(|e| e.contains_subquery()),
            Rollup(re) => re.for_subquery(),
            With(we) => we.was.iter().any(|e| e.expr.contains_subquery()),
            _ => false,
        }
    }

    pub fn return_type(&self) -> ReturnType {
        match self {
            Expression::Duration(de) => de.return_type(),
            Expression::Number(ne) => ne.return_type(),
            Expression::String(se) => se.return_type(),
            Expression::Parens(pe) => pe.return_type(),
            Expression::Function(fe) => fe.return_type(),
            Expression::Aggregation(ae) => ae.return_type(),
            Expression::BinaryOperator(be) => be.return_type(),
            Expression::With(we) => we.return_type(),
            Expression::Rollup(re) => re.return_type(),
            Expression::MetricExpression(me) => me.return_type(),
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            Expression::Duration(_) => "Duration",
            Expression::Number(_) => "Scalar",
            Expression::String(_) => "String",
            Expression::Parens(_) => "Group",
            Expression::Function(_) => "Function",
            Expression::Aggregation(_) => "Aggregation",
            Expression::BinaryOperator(_) => "Operator",
            Expression::With(_) => "With expression",
            Expression::Rollup(_) => "Rollup",
            Expression::MetricExpression(_) => "Selector",
        }
    }

    pub fn is_metric_expression(&self) -> bool {
        match self {
            Expression::MetricExpression(_) => true,
            _ => false,
        }
    }

    pub fn is_binary_op(&self) -> bool {
        match self {
            Expression::BinaryOperator(_) => true,
            _ => false,
        }
    }

    pub fn adjust_comp_ops(&mut self) {
        visit_all(self, |expr: &mut Expression| match expr {
            Expression::BinaryOperator(be) => {
                let _ = be.adjust_comparison_op();
            }
            _ => {}
        });
    }

    /// returns a scalar expression
    pub fn scalar(value: f64) -> Expression {
        Expression::from(value)
    }

    /// returns a string literal expression
    pub fn string_literal(value: &str) -> Expression {
        Expression::from(value)
    }

    /// Return `self == other`
    pub fn eq(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Eql, other)
    }

    /// Return `self != other`
    pub fn not_eq(self, other: Expression) -> Expression {
        binary_expr(self, Operator::NotEq, other)
    }

    /// Return `self > other`
    pub fn gt(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Gt, other)
    }

    /// Return `self >= other`
    pub fn gt_eq(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Gte, other)
    }

    /// Return `self < other`
    pub fn lt(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Lt, other)
    }

    /// Return `self <= other`
    pub fn lt_eq(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Lte, other)
    }

    /// Return `self AND other`
    pub fn and(self, other: Expression) -> Expression {
        binary_expr(self, Operator::And, other)
    }

    /// Return `self OR other`
    pub fn or(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Or, other)
    }

    /// Calculate the modulus of two expressions.
    /// Return `self % other`
    pub fn modulus(self, other: Expression) -> Expression {
        binary_expr(self, Operator::Mod, other)
    }

    /// Return `self == bool NaN`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_NaN(self) -> Expression {
        self.eq(Expression::from(f64::NAN))
    }

    /// Return `self != bool NaN`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_NaN(self) -> Expression {
        self.not_eq(Expression::from(f64::NAN))
    }

    /// Return `self == bool 1`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_true(self) -> Expression {
        self.eq(Expression::from(1.0))
    }

    /// Return `self != BOOL 1`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_true(self) -> Expression {
        self.not_eq(Expression::from(1.0))
    }

    /// Return `self == bool 0`
    pub fn is_false(self) -> Expression {
        self.eq(Expression::from(1.0))
    }

    /// Return `self != bool 0`
    pub fn is_not_false(self) -> Expression {
        self.not_eq(Expression::from(0.0))
    }
}

impl ExpressionNode for Expression {
    fn cast(self) -> Expression {
        match self {
            Expression::Duration(d) => d.cast(),
            Expression::Number(n) => n.cast(),
            Expression::String(s) => s.cast(),
            Expression::BinaryOperator(b) => b.cast(),
            Expression::MetricExpression(m) => m.cast(),
            Expression::Parens(p) => p.cast(),
            Expression::Function(f) => f.cast(),
            Expression::Aggregation(a) => a.cast(),
            Expression::Rollup(r) => r.cast(),
            Expression::With(w) => w.cast(),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Expression::Duration(d) => write!(f, "{}", d)?,
            Expression::Number(n) => write!(f, "{}", n)?,
            Expression::String(s) => write!(f, "{}", s)?,
            Expression::BinaryOperator(be) => write!(f, "{}", be)?,
            Expression::MetricExpression(me) => write!(f, "{}", me)?,
            Expression::Parens(p) => write!(f, "{}", p)?,
            Expression::Function(func) => write!(f, "{}", func)?,
            Expression::Aggregation(a) => write!(f, "{}", a)?,
            Expression::Rollup(re) => write!(f, "{}", re)?,
            Expression::With(w) => write!(f, "{}", w)?,
        }
        Ok(())
    }
}

// crate private
impl Default for Expression {
    fn default() -> Self {
        Expression::from(1.0)
    }
}

impl From<f64> for Expression {
    fn from(v: f64) -> Self {
        Expression::Number(NumberExpr::from(v))
    }
}

impl From<i64> for Expression {
    fn from(v: i64) -> Self {
        Expression::Number(NumberExpr::from(v as f64))
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::Number(NumberExpr::from(value as f64))
    }
}

impl From<String> for Expression {
    fn from(s: String) -> Self {
        Expression::String(StringExpr::from(s))
    }
}

impl From<&str> for Expression {
    fn from(s: &str) -> Self {
        Expression::String(StringExpr::from(s))
    }
}

impl From<Vec<BExpression>> for Expression {
    fn from(list: Vec<BExpression>) -> Self {
        Expression::Parens(ParensExpr::new(list))
    }
}

impl From<Vec<Expression>> for Expression {
    fn from(list: Vec<Expression>) -> Self {
        let items = list
            .into_iter()
            .map(|x| Box::new(x))
            .collect::<Vec<BExpression>>();
        Expression::Parens(ParensExpr::new(items))
    }
}

fn binary_expr(left: Expression, op: Operator, right: Expression) -> Expression {
    let mut expr = BinaryExpr::new(op, left, right);
    expr.bool_modifier = op.is_comparison();
    Expression::BinaryOperator(expr)
}

impl ops::Add for Expression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Add, rhs)
    }
}

impl ops::Sub for Expression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Sub, rhs)
    }
}

impl ops::Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mul, rhs)
    }
}

impl ops::Div for Expression {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Div, rhs)
    }
}

impl ops::Rem for Expression {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mod, rhs)
    }
}

// todo: integrate checks from
// https://github.com/prometheus/prometheus/blob/fa6e05903fd3ce52e374a6e1bf4eb98c9f1f45a7/promql/parser/parse.go#L436
