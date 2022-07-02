use std::fmt;
use std::fmt::Display;
use std::string::{String, ToString};
use crate::lexer::{duration_value, escape_ident};
use crate::error::{Error, Result};
use crate::types::{BinaryOp, Group, LabelFilterOp};
use crate::types::expression_kind::ExpressionKind;
use crate::types::label_filter::{LabelFilter, LabelFilterExpr};

pub type Span = logos::Span;

pub trait Visitor<T> {
    fn visit_expr(&mut self, visitor: fn(e: &Expression) -> ());
}

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn kind(&self) -> ExpressionKind;
}

/// A root expression node.
///
/// These are all valid root expression types.
#[derive(PartialEq, Clone, Debug)]
pub enum Expression {
    Duration(DurationExpr),

    Number(NumberExpr),

    /// A single scalar string.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    String(StringExpr),

    /// A grouped expression wrapped in parentheses
    Group(Group),

    Parens(ParensExpr),

    /// A function call
    Function(FuncExpr),

    Aggregation(AggrFuncExpr),

    /// A binary operator expression that returns a boolean value
    BinaryOperator(BinaryOpExpr),

    With(WithExpr),

    Rollup(RollupExpr),

    MetricExpression(MetricExpr),
}

pub type BExpression = Box<Expression>;

impl Expression {
    pub fn is_scalar(&self) -> bool {
        match self {
            Expression::Duration(_) | Expression::Number(_) => true,
            Expression::Function(f) => {
                *f.is_scalar
            },
            _ => false,
        }
    }

    pub fn cast(node: impl ExpressionNode) -> Option<Self> {
        let result = match node.kind() {
            ExpressionKind::Binop => Self::BinaryOperator(BinaryOpExpr(node)),
            ExpressionKind::Number => Self::Number(NumberExpr(node)),
            ExpressionKind::Parens => Self::Parens(ParensExpr(node)),
            ExpressionKind::Function => Self::Function(FuncExpr(node)),
            ExpressionKind::Group => Self::Group(Group(node)),
            ExpressionKind::Aggregate => Self::Aggregation(AggrFuncExpr(node)),
            ExpressionKind::With => Self::With(WithExpr(node)),
            ExpressionKind::Metric => Self::MetricExpression(MetricExpr(node)),
            ExpressionKind::String => Self::String(StringExpr(node)),
            ExpressionKind::Duration => Self::Duration(DurationExpr(node)),
            ExpressionKind::Rollup => Self::Rollup(RollupExpr(node)),
        };

        Some(result)
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expression::Duration(d) => write!(f, "{}", d)?,
            Expression::Number(n) => write!(f, "{}", n)?,
            Expression::String(s) => write!(f, "{}", s)?,
            Expression::Group(g) => write!(f, "{}", g)?,
            Expression::BinaryOperator(be) => write!(f, "{}", be)?,
            Expression::MetricExpression(me) => write!(f, "{}", me)?,
            Expression::Parens(p) => write!(f, "{}", p)?,
            Expression::Function(f) => write!(f, "{}", f)?,
            Expression::Aggregation(a) => write!(f, "{}", a)?,
            Expression::Rollup(re) => write!(f, "{}", re)?,
            Expression::With(w) => write!(f, "{}", w)?,
            _ => {
                panic!("missing display implementation");
            }
        }
        Ok(())
    }
}

// StringExpr represents string expression.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringExpr {
    // S contains unquoted value for string expression.
    pub s: String,

    // Composite string has non-empty tokens.
    // They must be converted into S by expand_with_expr.
    // todo: SmallVec
    pub tokens: Option<Vec<String>>
}

impl StringExpr {
    pub fn new<S: Into<String>>(s: S) -> Self {
        StringExpr {
            s: s.into(),
            tokens: None
        }
    }

    pub fn len(&self) -> usize {
        return self.s.len()
    }

    #[inline]
    pub fn value(&self) -> &str {
        return &self.s;
    }
}

impl ExpressionNode for StringExpr {
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

// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct NumberExpr {
    // n is the parsed number, i.e. `1.23`, `-234`, etc.
    pub n: f64,
}

impl NumberExpr {
    pub fn new(v: f64) -> Self {
        NumberExpr { n: v }
    }

    #[inline]
    pub fn value(&self) -> f64 {
        *self.n
    }
}

impl ExpressionNode for NumberExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Number
    }
}

impl Display for NumberExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN")?;
        } else {
            if self.n.is_finite() {
                write!(f, "{}", self.n)?;
            } else {
                if self.n.is_sign_positive() {
                    write!(f, "+Inf")?;
                } else {
                    write!(f, "-Inf")?;
                }
            }
        }
        Ok(())
    }
}


// See https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
#[derive(Debug, Clone, PartialEq)]
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
    type Error = Error;

    fn try_from(op: &str) -> Result<Self> {
        use GroupModifierOp::*;

        match op.to_lowercase().as_str() {
            "on" => Ok(On),
            "ignoring" => Ok(Ignoring),
            _ => Err(Error::new("Unknown group_modifier op")),
        }
    }
}

/// An operator matching clause
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GroupModifier {
    pub op: GroupModifierOp,

    /// A list of labels to which the operator is applied
    pub labels: Vec<String>,

    pub span: Option<Span>
}

impl GroupModifier {
    pub fn new(op: GroupModifierOp) -> Self {
        GroupModifier {
            op,
            labels: vec![],
            span: None
        }
    }

    /// Creates a GroupModifier cause with the On operator
    pub fn on() -> Self {
        GroupModifier::new(GroupModifierOp::On)
    }

    /// Creates a GroupModifier clause using the Ignoring operator
    pub fn ignoring() -> Self {
        GroupModifier::new(GroupModifierOp::Ignoring)
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
    pub fn set_labels<I>(mut self, labels: I) -> Self
    where I: Iterator<Item:Into<String>>{
        self.labels = labels.iter().map(|l| (*l).to_string()).collect();
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
        if self.labels.len() > 0 {
            write!(f, " ")?;
            write_list(&self.labels, f, false);
        }
        if let Some(group) = &self.group {
            write!(f, " {}", group)?;
        }
        Ok(())
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum JoinModifierOp {
    GroupLeft,
    GroupRight,
}

impl Display for JoinModifierOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use JoinModifierOp::*;
        match self {
            GroupLeft => write!(f, "group_left")?,
            GroupRight => write!(f, "group_right"),
        }
        Ok(())
    }
}

impl TryFrom<&str> for JoinModifierOp {
    type Error = Error;

    fn try_from(op: &str) -> Result<Self> {
        use JoinModifierOp::*;

        match op.to_lowercase().as_str() {
            "group_left" => Ok(GroupLeft),
            "group_right" => Ok(GroupRight),
            _ => Err(Error::new("Unknown join_modifier op")),
        }
    }
}

/// A GroupModifier clause's nested grouping clause
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct JoinModifier {
    /// The GroupModifier group's operator type (left or right)
    pub op: JoinModifierOp,

    /// A list of labels to copy to the opposite side of the group operator, i.e.
    /// group_left(foo) copies the label `foo` from the right hand side
    pub labels: Vec<String>,

    pub span: Option<Span>
}

impl JoinModifier {
    pub fn new(op: JoinModifierOp) -> Self {
        JoinModifier {
            op,
            labels: vec![],
            span: None
        }
    }

    /// Creates a new JoinModifier with the Left op
    pub fn left() -> Self {
        JoinModifier::new(JoinModifierOp::GroupLeft)
    }

    /// Creates a new JoinModifier with the Right op
    pub fn right() -> Self {
        JoinModifier::new(JoinModifierOp::GroupLeft)
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
            write_labels(&self.labels, f);
        }
        Ok(())
    }
}

// BinaryOpExpr represents binary operation.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct BinaryOpExpr {
    // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: BinaryOp,

    // Bool indicates whether `bool` modifier is present.
    // For example, `foo >bool bar`.
    pub bool_modifier: bool,

    // GroupModifier contains modifier such as "on" or "ignoring".
    pub group_modifier: Option<GroupModifier>,

    // JoinModifier contains modifier such as "group_left" or "group_right".
    pub join_modifier: Option<JoinModifier>,

    // Left contains left arg for the `left op right` expression.
    pub left: BExpression,

    // Right contains right arg for the `left op right` expression.
    pub right: BExpression,

    pub span: Option<Span>
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
            span: None
        }
    }

    pub fn balance(mut self) -> Self {
        let is_binop = match self.left {
            Expression::BinaryOperator(_) => true,
            _ => false
        };
        if !is_binop {
            return self;
        }
        let lp = self.left.op.precedence();
        let rp = self.right.op.precedence();
        if rp < lp {
            return self;
        }
        if rp == lp && !self.op.is_right_associative() {
            return self
        }
        self.left = self.right;
        self.right = Box::new(Expression::BinaryOperator(self.balance()) );
        return self;
    }

    pub(crate) fn swap_operands(mut self) {
        let temp: BExpression = self.left;
        self.left = self.right;
        self.right = temp;
    }

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl Display for BinaryOpExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Op is the operation itself, i.e. `+`, `-`, `*`, etc.
        match &self.left {
            Expression::BinaryOperator(be) => {
                write!(f, "({})", *self.left)?;
            },
            _ => {
                write!(f, "{}", *self.left)?;
            }
        }
        write!(f, " {}", self.op)?;
        if self.bool_modifier {
            write!(f, " bool")?;
        }
        if self.group_modifier.is_some() {
            write!(f, " {}", *self.group_modifier)?;
        }
        if self.join_modifier.is_some() {
            write!(f, " {}", *self.join_modifier)?;
        }
        write!(f, " ")?;
        match &self.right {
            Expression::BinaryOperator(be) => {
                write!(f, "({})", *self.right)?;
            },
            _ => {
                write!(f, "{}", *self.right)?;
            }
        }
        Ok(())
    }
}

impl ExpressionNode for BinaryOpExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Binop
    }
}

// FuncExpr represents MetricsQL function such as `foo(...)`
#[derive(Default, Debug, Clone, PartialEq)]
pub struct FuncExpr {
    // Name is function name.
    pub name: String,

    // Args contains function args.
    // todo: SmallVec
    pub args: Vec<Expression>,

    // If KeepMetricNames is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    pub span: Option<Span>
}

impl FuncExpr {
    pub fn new<S: Into<String>>(name: S, args: Vec<Expression>) -> Self {

        // time() returns scalar in PromQL - see https://prometheus.io/docs/prometheus/latest/querying/functions/#time
        let lower = name.to_lowercase().as_string();
        let is_scalar = lower == "time";

        FuncExpr {
            name: name.into(),
            args,
            keep_metric_names: false,
            span: None,
            is_scalar
        }
    }

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }
}

impl Display for FuncExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        write_expression_list(&self.args, f);
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

impl ExpressionNode for FuncExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Function
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct RollupExpr {
    // The expression for the rollup. Usually it is MetricExpr, but may be arbitrary expr
    // if subquery is used. https://prometheus.io/blog/2019/01/28/subquery-support/
    pub expr: BExpression,

    // Window contains optional window value from square brackets
    //
    // For example, `http_requests_total[5m]` will have Window value `5m`.
    pub window: Option<DurationExpr>,

    // Offset contains optional value from `offset` part.
    //
    // For example, `foobar{baz="aa"} offset 5m` will have Offset value `5m`.
    pub offset: Option<DurationExpr>,

    // Step contains optional step value from square brackets.
    //
    // For example, `foobar[1h:3m]` will have Step value '3m'.
    pub step: Option<DurationExpr>,

    // If set to true, then `foo[1h:]` would print the same
    // instead of `foo[1h]`.
    pub inherit_step: bool,

    // At contains an optional expression after `@` modifier.
    //
    // For example, `foo @ end()` or `bar[5m] @ 12345`
    // See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
    pub at: Option<BExpression>,

    pub span: Option<Span>
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
            span: None
        }
    }

    pub fn for_subquery(&self) -> bool {
        return self.step.is_some() || self.inherit_step.is_some();
    }

    pub fn set_at(mut self, expr: Expression) -> Self {
        self.at = Some(Box::new(expr));
        self
    }

    pub fn set_offset(mut self, expr: DurationExpr) -> Self {
        self.offset = Some(expr);
        self
    }

    pub fn set_window(mut self, expr: DurationExpr) -> Self {
        self.window = Some(expr);
        self
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let need_parent = match self.expr {
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
        let Some(window) = &self.window;
        let Some(offset) = &self.offset;
        let Some(inherit_step) = &self.inherit_step;
        let Some(step) = &self.step;

        if window.is_some() || inherit_step.is_some() || step.is_some() {
            write!(f, "[")?;
            if window.is_some() {
                write!(f, "{:?}", self.window)?;
            }
            if step.is_some() {
                write!(f, ":{}", step)?;
            } else if inherit_step.is_some() {
                write!(f, ":")?;
            }
            write!(f, "]")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " offset {}", offset)?;
        }
        if let Some(at) = &self.at {
            let parens_needed = match at {
                Expression::BinaryOperator(..) => true,
                _ => false,
            };
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
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Rollup
    }
}

// DurationExpr contains the duration
#[derive(Default, Debug, Clone, PartialEq)]
pub struct DurationExpr {
    pub s: String,
    pub span: Option<Span>
}

impl DurationExpr {
    pub fn new<S : Into<String>>(s: S) -> DurationExpr {
        DurationExpr {
            s: s.into(),
            span: None
        }
    }

    // Duration returns the duration from de in milliseconds.
    pub fn duration(&self, step: i64) -> Result<i64> {
        let d = match duration_value(&self.s, step) {
            Ok(d) => d,
            Err(e) => return Err(Error::new(format!("BUG: cannot parse duration {}: {}", self.s, e)))
        };
        Ok(d)
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.s)?;
        Ok(())
    }
}

impl ExpressionNode for RollupExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Duration
    }
}

// withExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, PartialEq)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>, // todo: SmallVec
    pub expr: BExpression,
    pub span: Option<Span>
}

impl WithExpr {
    pub fn new(expr: Expression, was: Vec<WithArgExpr>) -> Self {
        WithExpr {
            expr: Box::new(expr),
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
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::With
    }
}
// withArgExpr represents a single entry from WITH expression.
//
#[derive(Default, Debug, Clone, PartialEq)]
pub(crate) struct WithArgExpr {
    pub name: String,
    pub args: Vec<String>,
    pub expr: BExpression
}

impl WithArgExpr {
    pub fn new<S: Into<String>>(name: S, expr: Expression) -> Self {
        WithArgExpr {
            name: name.into(),
            args: vec![],
            expr: Box::new(expr)
        }
    }
}

impl Display for WithArgExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.name))?;
        write_list(&self.args, f, self.args.len() > 0);
        write!(f, " = ")?;
        write!(f, "{}", self.expr)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateModifierOp {
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
    type Error = Error;

    fn try_from(op: &str) -> Result<Self> {
        use AggregateModifierOp::*;

        match op.to_lowercase().as_str() {
            "by" => Ok(By),
            "without" => Ok(Without),
            _ => Err(Error::new("Unknown aggregate modifier op")),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct AggregateModifier {
    // Op is modifier operation.
    pub op: AggregateModifierOp,
    // Args contains modifier args from parens.
    pub args: Vec<String>,
    pub span: Option<Span>
}

impl AggregateModifier {
    pub fn new(op: AggregateModifierOp) -> Self {
        AggregateModifier {
            op,
            args: Vec::new(),
            span: None
        }
    }

    /// Creates a new AggregateModifier with the Left op
    pub fn by() -> Self {
        AggregateModifier::new(AggregateModifierOp::By)
    }

    /// Creates a new AggregateModifier with the Right op
    pub fn without() -> Self {
        AggregateModifier::new(AggregateModifierOp::Without)
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
        write_labels(&self.args, f);
        Ok(())
    }
}

// AggrFuncExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, PartialEq)]
pub struct AggrFuncExpr {
    // Name is the function name.
    pub name: String,

    // Args is the function args.
    pub args: Vec<Expression>,

    // Modifier is optional modifier such as `by (...)` or `without (...)`.
    pub modifier: Option<AggregateModifier>,

    // Optional limit for the number of output time series.
    // This is MetricsQL extension.
    //
    // Example: `sum(...) by (...) limit 10` would return maximum 10 time series.
    pub limit: usize,

    pub keep_metric_names: bool,

    pub span: Option<Span>
}

impl AggrFuncExpr {
    pub fn new(name: &str) -> AggrFuncExpr {
        AggrFuncExpr {
            name: name.to_string(),
            args: vec![],
            modifier: None,
            limit: 0,
            keep_metric_names: false,
            span: None
        }
    }

    pub fn with_modifier(mut self, modifier: AggregateModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    pub fn with_args(mut self, args: &[Expression]) -> Self {
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
        match &self.args[0] {
            Expression::Function(fe) => {
                self.keep_metric_names = *fe.keep_metric_names;
            }
            _ => self.keep_metric_names = false
        }
    }
}

impl Display for AggrFuncExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.name))?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_expression_list(&self.args, f);
        }
        if let Some(modifier) = &self.modifier {
            write!(f, "{}", modifier)?;
        }
        if self.limit > 0 {
            write!(f, " limit {}", self.limit.as_str())?;
        }
        Ok(())
    }
}

impl ExpressionNode for AggrFuncExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Aggregate
    }
}

// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct MetricExpr {
    // LabelFilters contains a list of label filters from curly braces.
    // Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilter>,

    // label_filters must be expanded to LabelFilters by expand_with_expr.
    pub(crate) label_filter_exprs: Vec<LabelFilterExpr>,

    pub span: Option<Span>
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilter::new(LabelFilterOp::Equal,  "__name__", name).unwrap();
        MetricExpr {
            label_filters: vec![name_filter],
            label_filter_exprs: vec![],
            span: None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.label_filters.len() == 0 && self.label_filter_exprs.len() == 0
    }

    pub fn has_non_empty_metric_group(&self) -> bool {
        if self.label_filters.len() == 0 {
            return false
        }
        return self.label_filters[0].is_metric_name_filter()
    }

    pub fn is_only_metric_group(&self) -> bool {
        if self.has_non_empty_metric_group() {
            return false
        }
        return self.label_filters.len() == 1
    }

    pub fn metric_group(&self) -> Option<&str> {
        if self.label_filters.len() == 0 {
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
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut lfs: &[LabelFilter] = &self.label_filters;
        if lfs.len() > 0 {
            let lf = &lfs[0];
            if lf.label == "__name__" && !lf.is_negative && !lf.is_regexExp {
                write!(f, "{}", escape_ident(&lf.label))?;
                lfs = &lfs[1..];
            }
        }
        if lfs.len() > 0 {
            write!(f, "{{")?;
            for (i, lf) in lfs {
                write!(f, "{}", lf)?;
                if (i+1) < lfs.len() {
                    write!(f, ", ")
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
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Metric
    }
}

/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone)]
pub struct ParensExpr {
    pub expressions: Vec<Expression>,
    pub span: Option<Span>
}

impl ParensExpr {
    pub fn new(expressions: Vec<Expression>) -> Self {
        ParensExpr { expressions, span: None }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }
}

impl Display for ParensExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_expression_list(&self.expressions, f);
        Ok(())
    }
}

impl ExpressionNode for ParensExpr {
    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Parens
    }
}

fn write_labels(labels: &[String], f: &mut fmt::Formatter) {
    if labels.len() > 0 {
        write!(f, "(")?;
        for (i, label) in labels.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", escape_ident(label))?;
        }
        write!(f, ")")?;
    }
}

fn write_expression_list(args: &[Expression], f: &mut fmt::Formatter) {
    write!(f, "(")?;
    for (i, arg) in args.iter().enumerate() {
        if (i + 1) < args.len() {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    write!(f, ")")?;
}

fn expr_list_to_str(args: &[Expression]) -> &String {
    let mut s = String::new();
    for (i, arg) in args.iter().enumerate() {
        if (i + 1) < args.len() {
            s.push_str(", ");
        }
        s.push_str(arg.as_str());
    }
    return &s;
}

fn write_list<I>(iterable: I, f: &mut fmt::Formatter, use_parens: bool)
    where I: IntoIterator,
          I::Item: Display
{
    let str = iterable.into_iter().join(", ");
    if use_parens {
        write!(f, "(")?;
    }
    write!(f, "{}", str)?;
    if use_parens {
        write!(f, ")")?;
    }
}


