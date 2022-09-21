use std::{fmt, iter};
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::string::{String, ToString};

use text_size::TextRange;

use lib::hash_f64;

use crate::ast::{BinaryOp, LabelFilterOp};
use crate::ast::expression_kind::ExpressionKind;
use crate::ast::label_filter::{LabelFilter, LabelFilterExpr};
use crate::ast::return_type::ReturnValue;
use crate::functions::{AggregateFunction, BuiltinFunction, get_aggregate_arg_idx_for_optimization, is_rollup_aggregation_over_time, TransformFunction};
use crate::lexer::{duration_value, escape_ident};
use crate::parser::{ParseError, ParseResult};

pub type Span = TextRange;

pub trait Visitor<T> {
    fn visit_expr(&mut self, visitor: fn(e: &Expression) -> ());
}

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn cast(self) -> Expression;
    fn kind(&self) -> ExpressionKind;
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, Hash)]
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

    Aggregation(AggrFuncExpr),

    /// A binary operator expression
    BinaryOperator(BinaryOpExpr),

    /// A Metricsql specific WITH statement node. Transformed at parse time to one
    /// of the other variants
    With(WithExpr),

    Rollup(RollupExpr),

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

    pub fn vectors(&self) -> Box<dyn Iterator<Item = &LabelFilter> + '_> {

        match self {
            Self::MetricExpression(v) => {
                Box::new(v.label_filters.iter() )
            },
            Self::Rollup(re) => {
                Box::new(
                    re.expr.vectors().chain(
                        if let Some(at) = &re.at {
                                at.vectors()
                              } else {
                                Box::new(iter::empty())
                              }
                        )
                )
            },
            Self::BinaryOperator(be) => {
                Box::new(be.left.vectors().chain(be.right.vectors()))
            },
            Self::Aggregation(ae) => {
                Box::new(ae.args.iter().flat_map(|node| node.vectors()))
            },
            Self::Function(fe) => {
                Box::new(fe.args.iter().flat_map(|node| node.vectors()))
            },
            Self::Parens(pe) => {
                Box::new(pe.expressions.iter().flat_map(|node| node.vectors()))
            },
            Self::Number(_) |
            Self::String(_) |
            Self::Duration(_) |
            Self::With(_) => Box::new(iter::empty()),
        }
    }

    /**
    Return an iterator of series names present in this node.
    ```
    let opts = promql::ParserOptions::new()
    	.allow_periods(false)
    	.build();
    let query = r#"
    	sum(1 - something_used{env="production"} / something_total) by (instance)
    	and ignoring (instance)
    	sum(rate(some_queries{instance=~"localhost\\d+"} [5m])) > 100
    "#;
    let ast = promql::parse(query, opts).expect("valid query");
    let series: Vec<String> = ast.series_names().collect();
    assert_eq!(series, vec![
    		"something_used".to_string(),
    		"something_total".to_string(),
    		"some_queries".to_string(),
    	],
    );
    ```
     */
    pub fn series_names(&self) -> impl Iterator<Item = String> + '_ {
        self.vectors().map(|x| {
            if x.label == "__name__" {
                x.value.clone()
                // String::from_utf8(x.value.clone())
                //     .expect("series names should always be valid utf8")
            } else {
                x.label.clone()
            }
        })
    }

    pub fn contains_subquery(&self) -> bool {
        use Expression::*;
        match self {
            Function(fe) => {
                fe.args.iter().any(|e| e.contains_subquery())
            }
            BinaryOperator(bo) => {
                bo.left.contains_subquery() || bo.right.contains_subquery()
            }
            Aggregation(aggr) => {
                aggr.args.iter().any(|e| e.contains_subquery())
            }
            Rollup(re) => {
                re.for_subquery()
            }
            _ => false
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        match self {
            Expression::Duration(de) => de.return_value(),
            Expression::Number(ne) => ne.return_value(),
            Expression::String(se) => se.return_value(),
            Expression::Parens(pe) => pe.return_value(),
            Expression::Function(fe) => fe.return_value(),
            Expression::Aggregation(ae) => ae.return_value(),
            Expression::BinaryOperator(be) => be.return_value(),
            Expression::With(_) => {
                // Todo
                ReturnValue::RangeVector
            }
            Expression::Rollup(re) => re.return_value(),
            Expression::MetricExpression(me) => me.return_value()
        }
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

    fn kind(&self) -> ExpressionKind {
        match self {
            Expression::Duration(..) => ExpressionKind::Duration,
            Expression::Number(..) => ExpressionKind::Number,
            Expression::String(..) => ExpressionKind::String,
            Expression::BinaryOperator(..) => ExpressionKind::Binop,
            Expression::MetricExpression(..) => ExpressionKind::Metric,
            Expression::Parens(..) => ExpressionKind::Parens,
            Expression::Function(..) => ExpressionKind::Function,
            Expression::Aggregation(..) => ExpressionKind::Aggregate,
            Expression::Rollup(..) => ExpressionKind::Rollup,
            Expression::With(..) => ExpressionKind::With,
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

impl From<f64> for Expression {
    fn from(v: f64) -> Self {
        Expression::Number(NumberExpr::new(v))
    }
}

impl From<i64> for Expression {
    fn from(v: i64) -> Self {
        Expression::Number(NumberExpr::new(v as f64))
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::Number(NumberExpr::new(value as f64))
    }
}

impl From<String> for Expression {
    fn from(s: String) -> Self {
        Expression::String(StringExpr::new(s))
    }
}

impl From<&str> for Expression {
    fn from(s: &str) -> Self {
        Expression::String (StringExpr::new(s))
    }
}

/// StringExpr represents string expression.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
pub struct StringExpr {
    /// S contains unquoted value for string expression.
    pub s: String,

    /// A composite string has non-empty tokens.
    /// They must be converted into S by expand_with_expr.
    // todo: SmallVec
    pub(crate) tokens: Option<Vec<String>>,
}

impl StringExpr {
    pub fn new<S: Into<String>>(s: S) -> Self {
        StringExpr {
            s: s.into(),
            tokens: None,
        }
    }

    pub fn len(&self) -> usize {
        self.s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    #[inline]
    pub fn value(&self) -> &str {
        &self.s
    }

    pub fn has_tokens(&self) -> bool {
        self.token_count() > 0
    }

    pub fn token_count(&self) -> usize {
        match &self.tokens {
            Some(v) => v.len(),
            None => 0,
        }
    }

    pub(crate) fn is_expanded(&self) -> bool {
        !self.s.is_empty() || self.token_count() > 0
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::String
    }
}

impl From<String> for StringExpr {
    fn from(s: String) -> Self {
        StringExpr::new(s)
    }
}

impl From<&str> for StringExpr {
    fn from(s: &str) -> Self {
        StringExpr::new(s)
    }
}

impl ExpressionNode for StringExpr {
    fn cast(self) -> Expression {
        Expression::String(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::String
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", enquote::enquote('"', &*self.s))?;
        Ok(())
    }
}

impl Deref for StringExpr {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Copy)]
pub struct NumberExpr {
    /// n is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
}

impl NumberExpr {
    pub fn new(v: f64) -> Self {
        NumberExpr { value: v }
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::Scalar
    }
}

impl From<f64> for NumberExpr {
    fn from(value: f64) -> Self {
        NumberExpr::new(value)
    }
}

impl From<i64> for NumberExpr {
    fn from(value: i64) -> Self {
        NumberExpr::new(value as f64)
    }
}

impl From<usize> for NumberExpr {
    fn from(value: usize) -> Self {
        NumberExpr::new(value as f64)
    }
}

impl Into<f64> for NumberExpr {
    fn into(self) -> f64 {
        self.value
    }
}

impl PartialEq for NumberExpr {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() <= f64::EPSILON
    }
}

impl ExpressionNode for NumberExpr {
    fn cast(self) -> Expression {
        Expression::Number(self)
    }
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Number
    }
}

impl Display for NumberExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.value.is_nan() {
            write!(f, "NaN")?;
        } else if self.value.is_finite() {
            write!(f, "{}", self.value)?;
        } else if self.value.is_sign_positive() {
            write!(f, "+Inf")?;
        } else {
            write!(f, "-Inf")?;
        }
        Ok(())
    }
}

impl Hash for NumberExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}

// See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum GroupModifierOp {
    On,
    Ignoring,
}

impl Display for GroupModifierOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct GroupModifier {
    /// Action applied to a list of vectors; whether `on (…)` or `ignored(…)` is used after the operator.
    pub op: GroupModifierOp,

    /// A list of labels to which the operator is applied
    pub labels: Vec<String>,

    pub span: Option<Span>,
}

impl GroupModifier {
    pub fn new(op: GroupModifierOp, labels: Vec<String>) -> Self {
        GroupModifier {
            op,
            labels,
            span: None,
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
    pub fn set_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Clears this GroupModifier's set of labels
    pub fn clear_labels(mut self) -> Self {
        self.labels.clear();
        self
    }

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl Display for GroupModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.op)?;
        if !self.labels.is_empty() {
            write!(f, " ")?;
            write_list(&self.labels, f, false)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash)]
pub enum JoinModifierOp {
    GroupLeft,
    GroupRight,
}

impl Display for JoinModifierOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

/// A GroupModifier clause's nested grouping clause
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct JoinModifier {
    /// The GroupModifier group's operator type (left or right)
    pub op: JoinModifierOp,

    /// A list of labels to copy to the opposite side of the group operator, i.e.
    /// group_left(foo) copies the label `foo` from the right hand side
    pub labels: Vec<String>,

    pub span: Option<Span>,
}

impl JoinModifier {
    pub fn new(op: JoinModifierOp) -> Self {
        JoinModifier {
            op,
            labels: vec![],
            span: None,
        }
    }

    /// Creates a new JoinModifier with the Left op
    pub fn left() -> Self {
        JoinModifier::new(JoinModifierOp::GroupLeft)
    }

    /// Creates a new JoinModifier with the Right op
    pub fn right() -> Self {
        JoinModifier::new(JoinModifierOp::GroupRight)
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

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl Display for JoinModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.op)?;
        if !self.labels.is_empty() {
            write!(f, " ")?;
            write_labels(&self.labels, f)?;
        }
        Ok(())
    }
}

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, Hash)]
pub struct BinaryOpExpr {
    /// Op is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: BinaryOp,

    /// Bool indicates whether `bool` modifier is present.
    /// For example, `foo >bool bar`.
    pub bool_modifier: bool,

    /// GroupModifier contains modifier such as "on" or "ignoring".
    pub group_modifier: Option<GroupModifier>,

    /// JoinModifier contains modifier such as "group_left" or "group_right".
    pub join_modifier: Option<JoinModifier>,

    /// Left contains left arg for the `left op right` expression.
    pub left: BExpression,

    /// Right contains right arg for the `left op right` expression.
    pub right: BExpression,

    pub span: Span,
}

impl BinaryOpExpr {
    pub fn new(op: BinaryOp, lhs: Expression, rhs: Expression) -> Self {
        BinaryOpExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            join_modifier: None,
            group_modifier: None,
            bool_modifier: false,
            span: Span::default(),
        }
    }

    /// Unary minus. Substitute `-expr` with `0 - expr`
    pub fn new_unary_minus(e: impl ExpressionNode) -> Self {
        let expr = Expression::cast(e);
        let lhs = Expression::Number(NumberExpr::new(0.0));
        BinaryOpExpr::new(BinaryOp::Sub, lhs, expr)
    }

    pub fn group_modifier_op_or_default(&self) -> GroupModifierOp {
        match &self.group_modifier {
            None => GroupModifierOp::Ignoring,
            Some(modifier) => modifier.op,
        }
    }

    /// Convert 'num cmpOp query' expression to `query reverseCmpOp num` expression
    /// like Prometheus does. For instance, `0.5 < foo` must be converted to `foo > 0.5`
    /// in order to return valid values for `foo` that are bigger than 0.5.
    pub fn adjust_comparison_op(&mut self) -> bool {
        if !self.op.is_comparison() {
            return false
        }
        if Expression::is_number(&self.right) || !Expression::is_scalar(&self.left) {
            return false
        }

        self.op = self.op.get_reverse_cmp();
        std::mem::swap(&mut self.left, &mut self.right);
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
            (ReturnValue::String, ReturnValue::String) => {
                if self.op != BinaryOp::Add {
                    return ReturnValue::unknown(
                        format!("Operator {} is not valid for (String, String)", &self.op),
                        self.clone().cast()
                    );
                }
                return ReturnValue::String
            },
            _ => return ReturnValue::InstantVector
        }
    }
}

impl Display for BinaryOpExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
        match &*self.left {
            Expression::BinaryOperator(be) => {
                write!(f, "({})", *be)?;
            }
            _ => {
                write!(f, "{}", *self.left)?;
            }
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
        match *self.right {
            Expression::BinaryOperator(_) => {
                write!(f, "({})", self.right)?;
            }
            _ => {
                write!(f, "{}", self.right)?;
            }
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

/// FuncExpr represents MetricsQL function such as `foo(...)`
#[derive(Debug, Clone, Hash)]
pub struct FuncExpr {
    pub function: BuiltinFunction,

    /// Args contains function args.
    pub args: Vec<BExpression>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    pub span: Option<Span>,

    /// internal only name parsed in WITH expression
    pub(crate) with_name: String,
}

impl FuncExpr {
    pub fn new(name: &str, args: Vec<BExpression>) -> ParseResult<Self> {
        // time() returns scalar in PromQL - see https://prometheus.io/docs/prometheus/latest/querying/functions/#time
        let lower = name.to_lowercase();
        let function = BuiltinFunction::new(name)?;
        let is_scalar = lower == "time";

        Ok(FuncExpr {
            function,
            args,
            keep_metric_names: false,
            span: None,
            is_scalar,
            with_name: "".to_string()
        })
    }

    pub fn name(&self) -> String {
        self.function.name()
    }

    pub fn default_rollup(arg: Expression) -> ParseResult<Self> {
        FuncExpr::from_single_arg("default_rollup", arg)
    }

    pub fn from_single_arg(name: &str, arg: Expression) -> ParseResult<Self> {
        let args = vec![Box::new(arg)];
        FuncExpr::new(name, args)
    }

    pub fn create(name: &str, args: &[Expression]) -> ParseResult<Self> {
        let params =
            Vec::from(args).into_iter().map(Box::new).collect();
        FuncExpr::new(name, params)
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = Some(span.into());
        self
    }

    pub fn is_aggregate(&self) -> bool {
        self.type_name() == "aggregate"
    }

    pub fn is_rollup(&self) -> bool {
        self.type_name() == "rollup"
    }

    pub fn type_name(&self) -> &'static str {
        self.function.type_name()
    }

    pub fn return_value(&self) -> ReturnValue {
        // functions generally pass through labels of one of their arguments, with
        // some exceptions

        if self.is_scalar {
            return ReturnValue::Scalar
        }

        // determine the arg to pass through
        let arg = self.get_arg_for_optimization();

        let kind = if arg.is_none() {
            ReturnValue::RangeVector
        } else {
            arg.unwrap().return_value()
        };

        return match self.function {
            BuiltinFunction::Rollup(rf) => {
                if is_rollup_aggregation_over_time(rf) {
                    match kind {
                        ReturnValue::RangeVector => ReturnValue::InstantVector,
                        _ => {
                            // invalid arg
                            ReturnValue::unknown(
                                format!("aggregation over time is not valid with expression returning {:?}", kind),
                                // show the arg as the cause
                                // doesn't follow the usual pattern of showing the parent, but would
                                // otherwise be ambiguous with multiple args
                                *arg.unwrap().clone()
                            )
                        }
                    }
                } else {
                    kind
                }
            },
            _ => kind
        }
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        self.function.get_arg_idx_for_optimization(&self.args)
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ BExpression> {
        match self.get_arg_idx_for_optimization() {
            None => None,
            Some(idx) => {
                Some(&self.args[idx])
            }
        }
    }
}

impl Display for FuncExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name())?;
        write_expression_list(&self.args, f)?;
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

impl ExpressionNode for FuncExpr {
    fn cast(self) -> Expression {
        Expression::Function(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Function
    }
}

#[derive(Debug, Clone, Hash)]
/// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
pub struct RollupExpr {
    /// The expression for the rollup. Usually it is MetricExpr, but may be arbitrary expr
    /// if subquery is used. https://prometheus.io/blog/2019/01/28/subquery-support/
    pub expr: BExpression,

    /// window contains optional window value from square brackets. Equivalent to `range` in
    /// prometheus terminology
    ///
    /// For example, `http_requests_total[5m]` will have Window value `5m`.
    pub window: Option<DurationExpr>,

    /// step contains optional step value from square brackets. Equivalent to `resolution`
    /// in the prometheus docs
    ///
    /// For example, `foobar[1h:3m]` will have step value `3m`.
    pub step: Option<DurationExpr>,

    /// offset contains optional value from `offset` part.
    ///
    /// For example, `foobar{baz="aa"} offset 5m` will have Offset value `5m`.
    pub offset: Option<DurationExpr>,

    /// if set to true, then `foo[1h:]` would print the same instead of `foo[1h]`.
    pub inherit_step: bool,

    /// at contains an optional expression after `@` modifier.
    ///
    /// For example, `foo @ end()` or `bar[5m] @ 12345`
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
    pub at: Option<BExpression>,

    pub span: Option<Span>,
}

impl RollupExpr {
    pub fn new(expr: Expression) -> Self {
        RollupExpr {
            expr: Box::new(expr),
            window: None,
            offset: None,
            step: None,
            inherit_step: false,
            at: None,
            span: None,
        }
    }

    pub fn for_subquery(&self) -> bool {
        self.step.is_some() || self.inherit_step
    }

    pub fn set_at(mut self, expr: impl ExpressionNode) -> Self {
        self.at = Some(Box::new(expr.cast()));
        self
    }

    pub fn set_offset(&mut self, expr: DurationExpr) {
        self.offset = Some(expr);
    }

    pub fn set_window(mut self, expr: DurationExpr) -> Self {
        self.window = Some(expr);
        self
    }

    pub fn set_expr(&mut self, expr: impl ExpressionNode) {
        self.expr = Box::new(Expression::cast(expr));
    }

    pub fn return_value(&self) -> ReturnValue {
        // sub queries turn instant vectors into ranges
        let kind = match (self.window.is_some(), self.for_subquery()) {
            (false, false) => ReturnValue::InstantVector,
            (false, true) => ReturnValue::RangeVector,
            (true, false) => ReturnValue::RangeVector,

            // range + subquery is not allowed (however this is syntactically invalid)
            (true, true) => ReturnValue::unknown(
                "range and subquery are not allowed together",
                self.clone().cast()
            )
        };

        kind
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let need_parent = match *self.expr {
            Expression::Rollup(..) => false,
            Expression::BinaryOperator(..) => true,
            Expression::Aggregation(..) => true,
            _ => false,
        };
        if need_parent {
            write!(f, "(")?;
        }
        write!(f, "{}", self.expr)?;
        if need_parent {
            write!(f, ")")?;
        }

        if self.window.is_some() || self.inherit_step || self.step.is_some() {
            write!(f, "[")?;
            if self.window.is_some() {
                write!(f, "{:?}", self.window.as_ref().unwrap())?;
            }
            if self.step.is_some() {
                write!(f, ":{}", self.step.as_ref().unwrap())?;
            } else if self.inherit_step {
                write!(f, ":")?;
            }
            write!(f, "]")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " offset {}", offset)?;
        }
        if let Some(at) = &self.at {
            let parens_needed = self.kind() == ExpressionKind::Binop;
            write!(f, "@")?;
            if parens_needed {
                write!(f, "(")?;
            }
            write!(f, "{}", at)?;
            if parens_needed {
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

impl ExpressionNode for RollupExpr {
    fn cast(self) -> Expression {
        Expression::Rollup(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Rollup
    }
}

/// DurationExpr contains the duration
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct DurationExpr {
    pub s: String,
    pub span: Span,
    pub const_value: i64,
    pub requires_step: bool,
}

impl TryFrom<&str> for DurationExpr {
    type Error = ParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let last_ch: char = s.chars().rev().next().unwrap();
        let const_value = duration_value(&s, 1)?;
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';

        Ok(Self {
            s: s.to_string(),
            const_value,
            requires_step,
            span: Span::default(),
        })
    }
}


impl DurationExpr {
    pub fn new(s: String, span: Span) -> DurationExpr {
        let last_ch: char = s.chars().rev().next().unwrap();
        let requires_step: bool = last_ch == 'i' || last_ch == 'I';
        // todo: the following is icky
        let const_value = duration_value(&s, 1).unwrap_or(0);

        DurationExpr {
            s,
            const_value,
            requires_step,
            span,
        }
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn duration(&self, step: i64) -> i64 {
        if self.requires_step {
            self.const_value * step
        } else {
            self.const_value
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::Scalar
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.s)?;
        Ok(())
    }
}

impl ExpressionNode for DurationExpr {
    fn cast(self) -> Expression {
        Expression::Duration(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Duration
    }
}

/// withExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, Hash)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>,
    pub expr: BExpression,
    pub span: Option<Span>,
}

impl WithExpr {
    pub fn new(expr: impl ExpressionNode, was: Vec<WithArgExpr>) -> Self {
        let expression = Expression::cast(expr);
        WithExpr {
            expr: Box::new(expression),
            was,
            span: None,
        }
    }
}

impl Display for WithExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.name))?;
        write_list(&self.args, f, !self.args.is_empty())?;
        write!(f, " = ")?;
        write!(f, "{}", self.expr)?;
        Ok(())
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AggregateModifierOp {
    #[default]
    By,
    Without,
}

impl Display for AggregateModifierOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AggregateModifierOp::*;
        match self {
            By => write!(f, "by")?,
            Without => write!(f, "without")?,
        }
        Ok(())
    }
}

impl TryFrom<&str> for AggregateModifierOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        use AggregateModifierOp::*;

        match op.to_lowercase().as_str() {
            "by" => Ok(By),
            "without" => Ok(Without),
            _ => Err(ParseError::General(format!(
                "Unknown aggregate modifier op: {}",
                op
            ))),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggregateModifier {
    /// The modifier operation.
    pub op: AggregateModifierOp,
    /// Modifier args from parens.
    pub args: Vec<String>,
    pub span: Option<Span>,
}

impl AggregateModifier {
    pub fn new(op: AggregateModifierOp, args: Vec<String>) -> Self {
        AggregateModifier {
            op,
            args,
            span: None,
        }
    }

    /// Creates a new AggregateModifier with the Left op
    pub fn by() -> Self {
        AggregateModifier::new(AggregateModifierOp::By, vec![])
    }

    /// Creates a new AggregateModifier with the Right op
    pub fn without() -> Self {
        AggregateModifier::new(AggregateModifierOp::Without, vec![])
    }

    /// Replaces this AggregateModifier's operator
    pub fn op(mut self, op: AggregateModifierOp) -> Self {
        self.op = op;
        self
    }

    /// Adds a label key to this AggregateModifier
    pub fn arg<S: Into<String>>(mut self, arg: S) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Replaces this AggregateModifier's args with the given set
    pub fn args(mut self, args: &[&str]) -> Self {
        self.args = args.iter().map(|l| (*l).to_string()).collect();
        self
    }
}

impl Display for AggregateModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
        write!(f, "{}", self.op)?;
        write_labels(&self.args, f)?;
        Ok(())
    }
}

/// AggrFuncExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, Hash)]
pub struct AggrFuncExpr {
    /// the aggregation function enum
    pub function: AggregateFunction,

    /// name is the function name.
    pub name: String,

    /// function args.
    pub args: Vec<BExpression>,

    /// optional modifier such as `by (...)` or `without (...)`.
    pub modifier: Option<AggregateModifier>,

    /// optional limit for the number of output time series.
    /// This is MetricsQL extension.
    ///
    /// Example: `sum(...) by (...) limit 10` would return maximum 10 time series.
    pub limit: usize,

    pub keep_metric_names: bool,

    pub span: Span,
}

impl AggrFuncExpr {
    pub fn new(function: &AggregateFunction) -> AggrFuncExpr {
        AggrFuncExpr {
            function: *function,
            name: function.to_string(),
            args: vec![],
            modifier: None,
            limit: 0,
            keep_metric_names: false,
            span: Span::default(),
        }
    }

    pub fn with_modifier(mut self, modifier: AggregateModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    pub fn with_args(mut self, args: &[BExpression]) -> Self {
        self.args = args.to_vec();
        self.set_keep_metric_names();
        self
    }

    fn set_keep_metric_names(&mut self) {
        // Extract: RollupFunc(...) from aggrFunc(rollupFunc(...)).
        // This case is possible when optimized aggrfn calculations are used
        // such as `sum(rate(...))`
        if self.args.len() != 1 {
            self.keep_metric_names = false;
            return;
        }
        match &*self.args[0] {
            Expression::Function(fe) => {
                self.keep_metric_names = fe.keep_metric_names;
            }
            _ => self.keep_metric_names = false,
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::InstantVector
    }

    pub fn get_arg_idx_for_optimization(&self) -> Option<usize> {
        get_aggregate_arg_idx_for_optimization(self.function, self.args.len())
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ BExpression> {
        match self.get_arg_idx_for_optimization() {
            None => None,
            Some(idx) => {
                Some(&self.args[idx])
            }
        }
    }
}

impl Display for AggrFuncExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.function)?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_expression_list(&self.args, f)?;
        }
        if let Some(modifier) = &self.modifier {
            write!(f, "{}", modifier)?;
        }
        if self.limit > 0 {
            write!(f, " limit {}", self.limit)?;
        }
        Ok(())
    }
}

impl ExpressionNode for AggrFuncExpr {
    fn cast(self) -> Expression {
        Expression::Aggregation(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Aggregate
    }
}

/// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Default, Debug, Clone, PartialEq, Hash)]
pub struct MetricExpr {
    /// LabelFilters contains a list of label filters from curly braces.
    /// Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilter>,

    /// label_filters must be expanded to LabelFilters by expand_with_expr.
    pub(crate) label_filter_exprs: Vec<LabelFilterExpr>,

    pub span: Option<Span>,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilter::new(LabelFilterOp::Equal, "__name__", name).unwrap();
        MetricExpr {
            label_filters: vec![name_filter],
            label_filter_exprs: vec![],
            span: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.label_filters.len() == 0 && self.label_filter_exprs.len() == 0
    }

    pub fn has_non_empty_metric_group(&self) -> bool {
        if self.label_filters.is_empty() {
            return false;
        }
        self.label_filters[0].is_metric_name_filter()
    }

    pub fn is_only_metric_group(&self) -> bool {
        if self.has_non_empty_metric_group() {
            return false;
        }
        self.label_filters.len() == 1
    }

    pub fn metric_group(&self) -> Option<&str> {
        if self.label_filters.is_empty() {
            None
        } else {
            let filter = &self.label_filters[0];
            if filter.label == "__name__" {
                Some(&filter.value)
            } else {
                None
            }
        }
    }

    pub fn return_value(&self) -> ReturnValue {
        //??? TODO:: is this correct ?
        ReturnValue::RangeVector
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut lfs: &[LabelFilter] = &self.label_filters;
        if !lfs.is_empty() {
            let lf = &lfs[0];
            if lf.label == "__name__" && !lf.is_negative() && !lf.is_regexp() {
                write!(f, "{}", escape_ident(&lf.label))?;
                lfs = &lfs[1..];
            }
        }
        if !lfs.is_empty() {
            write!(f, "{{")?;
            for (i, lf) in lfs.iter().enumerate() {
                write!(f, "{}", lf)?;
                if (i + 1) < lfs.len() {
                    write!(f, ", ")?;
                }
            }
            write!(f, "}}")?;
        } else {
            write!(f, "{{}}")?;
        }
        Ok(())
    }
}

impl ExpressionNode for MetricExpr {
    fn cast(self) -> Expression {
        Expression::MetricExpression(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Metric
    }
}

/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, Hash)]
pub struct ParensExpr {
    pub expressions: Vec<BExpression>,
    pub span: Option<Span>,
}

impl ParensExpr {
    pub fn new(expressions: Vec<BExpression>) -> Self {
        ParensExpr {
            expressions,
            span: None,
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_expression_list(&self.expressions, f)?;
        Ok(())
    }
}

impl ExpressionNode for ParensExpr {
    fn cast(self) -> Expression {
        Expression::Parens(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Parens
    }
}

fn write_labels(labels: &[String], f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    if !labels.is_empty() {
        write!(f, "(")?;
        for (i, label) in labels.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", escape_ident(label))?;
        }
        write!(f, ")")?;
    }
    Ok(())
}

fn write_expression_list(exprs: &[BExpression], f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    let mut items: Vec<String> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        items.push(format!("{}", expr));
    }
    write_list(&items, f, true)?;
    Ok(())
}

fn write_list(
    values: &Vec<String>,
    f: &mut fmt::Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.iter().enumerate() {
        if (i + 1) < values.len() {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    if use_parens {
        write!(f, ")")?;
    }
    Ok(())
}
