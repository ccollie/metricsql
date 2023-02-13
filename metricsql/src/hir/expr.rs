use std::{fmt, iter, ops};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use enquote::enquote;
use chrono::Duration;
use lib::{fmt_duration_ms, hash_f64};
use serde::{Serialize, Deserialize};
use crate::ast;
use crate::functions::{AggregateFunction, BuiltinFunction};
use crate::hir::{
    AggregateModifier,
    Operator,
    GroupModifier,
    JoinModifier,
    LabelFilter,
    LabelFilterOp,
    ReturnType,
    VectorMatchCardinality
};

// todo: number => scalar
/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct NumberExpr {
    /// value is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
}

impl NumberExpr {
    pub fn new(v: f64) -> Self {
        NumberExpr {
            value: v,
        }
    }
    pub fn return_type(&self) -> ReturnType {
        ReturnType::Scalar
    }
}

impl From<f64> for NumberExpr {
    fn from(value: f64) -> Self {
        NumberExpr::new(value)
    }
}

impl From<ast::NumberExpr> for NumberExpr {
    fn from(expr: ast::NumberExpr) -> Self {
        NumberExpr::from(expr.value)
    }
}

impl From<&ast::NumberExpr> for NumberExpr {
    fn from(expr: &ast::NumberExpr) -> Self {
        NumberExpr::from(expr.value)
    }
}

impl PartialEq for NumberExpr {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() <= f64::EPSILON
    }
}


impl Display for NumberExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", format_num(self.value))?;
        Ok(())
    }
}

impl Hash for NumberExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}


/// StringExpr represents string expression.
#[derive(Debug, Clone, Default, Hash, Serialize, Deserialize)]
pub struct StringExpr {
    /// contains unquoted value for string expression.
    pub value: String,
}

impl StringExpr {
    pub fn new<S: Into<String>>(s: S) -> Self {
        StringExpr {
            value: s.into()
        }
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::String
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

impl From<ast::StringExpr> for StringExpr {
    fn from(value: ast::StringExpr) -> Self {
        StringExpr::from(value.value)
    }
}

impl From<&ast::StringExpr> for StringExpr {
    fn from(value: &ast::StringExpr) -> Self {
        StringExpr::from(value.value.as_str())
    }
}

impl Display for StringExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", enquote('"', &*self.value))?;
        Ok(())
    }
}

impl Deref for StringExpr {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// DurationExpr contains a duration
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurationExpr {
    /// Duration in milliseconds
    pub value: i64,
    pub requires_step: bool,
}

impl From<&ast::DurationExpr>  for DurationExpr {
    fn from(value: &ast::DurationExpr) -> Self {
        Self {
            value: value.value,
            requires_step: value.requires_step,
        }
    }
}

impl DurationExpr {
    pub fn new(millis: i64, needs_step: bool) -> Self {
        Self {
            value: millis,
            requires_step: needs_step,
        }
    }

    /// Duration returns the duration from de in milliseconds.
    pub fn value(&self, step: i64) -> i64 {
        if self.requires_step {
            self.value * step
        } else {
            self.value
        }
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::Scalar
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        fmt_duration_ms(f, self.value)?;
        if self.requires_step {
            write!(f, "I")?;
        }
        Ok(())
    }
}

// todo: MetricExpr => Selector
/// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct MetricExpr {
    /// LabelFilters contains a list of label filters from curly braces.
    /// Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilter>,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilter::new(LabelFilterOp::Equal, "__name__", name).unwrap();
        MetricExpr {
            label_filters: vec![name_filter]
        }
    }

    pub fn with_filters(filters: Vec<LabelFilter>) -> Self {
        MetricExpr {
            label_filters: filters,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.label_filters.is_empty()
    }

    pub fn has_non_empty_metric_group(&self) -> bool {
        if self.label_filters.is_empty() {
            return false;
        }
        self.label_filters[0].is_metric_name_filter()
    }

    pub fn is_only_metric_group(&self) -> bool {
        if !self.has_non_empty_metric_group() {
            return false;
        }
        self.label_filters.len() == 1
    }

    pub fn name(&self) -> Option<&str> {
        match self.label_filters.iter().find(|filter| filter.label == "__name__") {
            Some(f) => Some(&f.value),
            None => None
        }
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::InstantVector
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut lfs: &[LabelFilter] = &self.label_filters;
        if !lfs.is_empty() {
            let lf = &lfs[0];
            if lf.label == "__name__" && !lf.is_negative() && !lf.is_regexp() {
                write!(f, "{}", &lf.value)?;
                lfs = &lfs[1..];
            }
        }
        if !lfs.is_empty() {
            write!(f, "{{")?;
            for (i, lf) in lfs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", lf)?;
            }
            write!(f, "}}")?;
        } else if self.label_filters.len() == 0 {
            write!(f, "{{}}")?;
        }
        Ok(())
    }
}

impl From<&ast::MetricExpr> for MetricExpr {
    fn from(value: &ast::MetricExpr) -> Self {
        let label_filters = value.label_filters.clone();
        MetricExpr {
            label_filters
        }
    }
}

impl Default for MetricExpr {
    fn default() -> Self {
        Self {
            label_filters: vec![],
        }
    }
}


// TODO: ParensExpr => GroupExpr
/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, Hash, Serialize, Deserialize)]
pub struct ParensExpr {
    pub expressions: Vec<BExpression>,
    pub return_type: ReturnType
}

impl ParensExpr {
    pub fn new(expressions: Vec<BExpression>) -> Self {
        ParensExpr {
            expressions,
            return_type: ReturnType::Scalar
        }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }

    pub fn return_type(&self) -> ReturnType {
        self.return_type.clone()
    }
}

impl From<&ast::ParensExpr> for ParensExpr {
    fn from(value: &ast::ParensExpr) -> Self {
        todo!()
    }
}

impl Display for ParensExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_expression_list(&self.expressions, f)?;
        Ok(())
    }
}

/// FuncExpr represents MetricsQL function such as `rate(...)`
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct FunctionExpr {
    pub name: String,

    pub function: BuiltinFunction,

    /// Args contains function args.
    pub args: Vec<BExpression>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    pub return_type: ReturnType,

    pub arg_idx_for_optimization: Option<usize>
}

impl FunctionExpr {

    pub fn is_aggregate(&self) -> bool {
        self.type_name() == "aggregate"
    }

    pub fn is_rollup(&self) -> bool {
        self.type_name() == "rollup"
    }

    pub fn type_name(&self) -> &'static str {
        self.function.type_name()
    }

    pub fn return_type(&self) -> ReturnType {
        self.return_type.clone()
    }
}

impl From<&ast::FuncExpr> for FunctionExpr {
    fn from(value: &ast::FuncExpr) -> Self {
        Self {
            name: value.name.clone(),
            function: value.function.clone(),
            args: clone_ast_expr_vec(&value.args),
            keep_metric_names: value.keep_metric_names,
            is_scalar: value.is_scalar,
            return_type: value.return_type(),
            arg_idx_for_optimization: value.get_arg_idx_for_optimization(),
        }
    }
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        write_expression_list(&self.args, f)?;
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}


/// AggregationExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct AggregationExpr {
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

    pub arg_idx_for_optimization: Option<usize>
}

impl AggregationExpr {
    pub fn new(function: &AggregateFunction) -> AggregationExpr {
        AggregationExpr {
            function: *function,
            name: function.to_string(),
            args: vec![],
            modifier: None,
            limit: 0,
            keep_metric_names: false,
            arg_idx_for_optimization: None,
        }
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::InstantVector
    }
}

impl From<&ast::AggrFuncExpr> for AggregationExpr {
    fn from(value: &ast::AggrFuncExpr) -> Self {
        Self {
            function: value.function.clone(),
            name: value.name.clone(),
            args: clone_ast_expr_vec(&value.args),
            modifier: value.modifier.clone(),
            limit: value.limit,
            keep_metric_names: value.keep_metric_names,
            arg_idx_for_optimization: value.get_arg_idx_for_optimization(),
        }
    }
}

impl Display for AggregationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.function)?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_expression_list(&self.args, f)?;
        }
        if let Some(modifier) = &self.modifier {
            write!(f, " {}", modifier)?;
        }
        if self.limit > 0 {
            write!(f, " limit {}", self.limit)?;
        }
        Ok(())
    }
}


#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
/// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
pub struct RollupExpr {
    /// The expression for the rollup. Usually it is MetricExpr, but may be arbitrary expr
    /// if subquery is used. https://prometheus.io/blog/2019/01/28/subquery-support/
    pub expr: BExpression,

    /// window contains optional window value from square brackets. Equivalent to `range` in
    /// prometheus terminology
    ///
    /// For example, `http_requests_total[5m]` will have Window value `5m`.
    #[serde_as(as = "DurationMilliseconds<i64>")]
    pub window: Option<Duration>,

    /// step contains optional step value from square brackets. Equivalent to `resolution`
    /// in the prometheus docs
    ///
    /// For example, `foobar[1h:3m]` will have step value `3m`.
    #[serde_as(as = "DurationMilliseconds<i64>")]
    pub step: Option<Duration>,

    /// offset contains optional value from `offset` part.
    ///
    /// For example, `foobar{baz="aa"} offset 5m` will have Offset value `5m`.
    #[serde_as(as = "DurationMilliseconds<i64>")]
    pub offset: Option<Duration>,

    /// if set to true, then `foo[1h:]` would print the same instead of `foo[1h]`.
    pub inherit_step: bool,

    /// at contains an optional expression after `@` modifier.
    ///
    /// For example, `foo @ end()` or `bar[5m] @ 12345`
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
    pub at: Option<BExpression>,

    _return_type: ReturnType
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
            _return_type: ReturnType::String
        }
    }

    pub fn for_subquery(&self) -> bool {
        self.step.is_some() || self.inherit_step
    }

    pub fn return_type(&self) -> ReturnType {
        self._return_type.clone()
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let need_parent = match *self.expr {
            Expression::Rollup(..) => true,
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
            if let Some(win) = &self.window {
                fmt_duration_ms(f, win.as_millis() as i64)?;
            }
            if let Some(step) = &self.step {
                fmt_duration_ms(f, step.as_millis() as i64)?;
            } else if self.inherit_step {
                write!(f, ":")?;
            }
            write!(f, "]")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " offset ")?;
            fmt_duration_ms(f, offset.as_millis() as i64)?;
        }
        if let Some(at) = &self.at {
            let parens_needed = at.is_binary_op();
            write!(f, " @ ")?;
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

impl From<&ast::RollupExpr> for RollupExpr {
    fn from(value: &ast::RollupExpr) -> Self {
        Self {
            expr: Box::new(Default::default()),
            window: None,
            step: None,
            offset: None,
            inherit_step: value.inherit_step,
            at: None,
            _return_type: value.return_type(),
        }
    }
}

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
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

    pub cardinality: VectorMatchCardinality,

    _return_type: ReturnType
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
            cardinality: VectorMatchCardinality::OneToOne,
            _return_type: ReturnType::Scalar,
        }
    }

    pub fn return_type(&self) -> ReturnType {
        self._return_type.clone()
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

impl From<&ast::BinaryExpr> for BinaryExpr {
    fn from(value: &ast::BinaryExpr) -> Self {
        Self {
            op: value.op,
            bool_modifier: value.bool_modifier,
            group_modifier: value.group_modifier.clone(),
            join_modifier: value.join_modifier.clone(),
            left: Box::new(Default::default()),  // TODO
            right: Box::new(Default::default()),
            cardinality: VectorMatchCardinality::OneToOne,
            _return_type: value.return_type(),
        }
    }
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, Hash, Serialize, Deserialize)]
pub enum Expression {
    /// A single scalar number.
    Number(NumberExpr),

    Duration(DurationExpr),

    /// A single scalar string.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    String(StringExpr),

    /// A grouped expression wrapped in parentheses
    Parens(ParensExpr),

    /// A function call
    Function(FunctionExpr),

    /// Aggregation represents aggregate functions such as `sum(...) by (...)`
    Aggregation(AggregationExpr),

    /// A binary operator expression
    BinaryOperator(BinaryExpr),

    /// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
    Rollup(RollupExpr),

    /// MetricExpr represents a MetricsQL metric with optional filters, i.e. `foo{...}`.
    MetricExpression(MetricExpr),
}

pub type BExpression = Box<Expression>;

impl Expression {
    pub fn is_scalar(expr: &Expression) -> bool {
        match expr {
            Expression::Duration(_) | Expression::Number(_) | Expression::Duration(_) => true,
            Expression::Function(f) => f.is_scalar,
            _ => false,
        }
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
            Self::Duration(_) |
            Self::String(_) => Box::new(iter::empty()),
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

    pub fn return_type(&self) -> ReturnType {
        match self {
            Expression::Number(ne) => ne.return_type(),
            Expression::Duration(dur) => dur.return_type(),
            Expression::String(se) => se.return_type(),
            Expression::Parens(pe) => pe.return_type(),
            Expression::Function(fe) => fe.return_type(),
            Expression::Aggregation(ae) => ae.return_type(),
            Expression::BinaryOperator(be) => be.return_type(),
            Expression::Rollup(re) => re.return_type(),
            Expression::MetricExpression(me) => me.return_type()
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            Expression::Number(_) => "Scalar",
            Expression::Duration(_) => "Scalar",
            Expression::String(_) => "String",
            Expression::Parens(_) => "Group",
            Expression::Function(_) => "Function",
            Expression::Aggregation(_) => "Aggregation",
            Expression::BinaryOperator(_) => "Operator",
            Expression::Rollup(_) => "Rollup",
            Expression::MetricExpression(_) => "Selector"
        }
    }

    pub fn is_metric_expression(&self) -> bool {
        match self {
            Expression::MetricExpression(_) => true,
            _ => false
        }
    }

    pub fn is_binary_op(&self) -> bool {
        match self {
            Expression::BinaryOperator(_) => true,
            _ => false
        }
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
        self.eq( Expression::from(f64::NAN) )
    }

    /// Return `self != bool NaN`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_NaN(self) -> Expression {
        self.not_eq(Expression::from(f64::NAN))
    }

    /// Return `self == bool 1`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_true(self) -> Expression {
        self.eq( Expression::from(1.0) )
    }

    /// Return `self != BOOL 1`
    #[allow(clippy::wrong_self_convention)]
    pub fn is_not_true(self) -> Expression {
        self.not_eq( Expression::from(1.0) )
    }

    /// Return `self == bool 0`
    pub fn is_false(self) -> Expression {
        self.eq(Expression::from(0.0))
    }

    /// Return `self != bool 0`
    pub fn is_not_false(self) -> Expression {
        self.not_eq(Expression::from(0.0))
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n)?,
            Expression::Duration(d) => write!(f, "{}", d)?,
            Expression::String(s) => write!(f, "{}", s)?,
            Expression::BinaryOperator(be) => write!(f, "{}", be)?,
            Expression::MetricExpression(me) => write!(f, "{}", me)?,
            Expression::Parens(p) => write!(f, "{}", p)?,
            Expression::Function(func) => write!(f, "{}", func)?,
            Expression::Aggregation(a) => write!(f, "{}", a)?,
            Expression::Rollup(re) => write!(f, "{}", re)?,
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
        Expression::String (StringExpr::from(s))
    }
}

impl From<&ast::Expression> for Expression {
    fn from(value: &ast::Expression) -> Self {
        match value {
            ast::Expression::Number(num) => Self::Number(NumberExpr::from(num)),
            ast::Expression::String(str) => Self::String(StringExpr::from(str)),
            ast::Expression::Duration(dur) => {
                if dur.requires_step {
                    Expression::Duration(DurationExpr::from(dur))
                } else {
                    Self::Number(NumberExpr::new(dur.value as f64))
                }
            },
            ast::Expression::Parens(parens) => Self::Parens(ParensExpr::from(parens)),
            ast::Expression::Function(func) => Self::Function(FunctionExpr::from(func)),
            ast::Expression::Aggregation(aggr) => Self::Aggregation(AggregationExpr::from(aggr)),
            ast::Expression::BinaryOperator(bexpr) => Self::BinaryOperator(BinaryExpr::from(bexpr)),
            ast::Expression::Rollup(rollup) => Self::Rollup(RollupExpr::from(rollup)),
            ast::Expression::MetricExpression(me) => Self::MetricExpression(MetricExpr::from(me)),
            ast::Expression::With(_) => {
                unreachable!()
            }
        }
    }
}

impl From<Vec<BExpression>> for Expression {
    fn from(list: Vec<BExpression>) -> Self {
        Expression::Parens(ParensExpr::new(list))
    }
}

impl From<Vec<Expression>> for Expression {
    fn from(list: Vec<Expression>) -> Self {
        let items = list.into_iter().map(|x| Box::new(x)).collect::<Vec<BExpression>>();
        Expression::Parens(ParensExpr::new(items))
    }
}

fn binary_expr(left: Expression, op: Operator, right: Expression) -> Expression {
    let mut expr = BinaryExpr::new(op,left, right);
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

fn write_expression_list(exprs: &[BExpression], f: &mut Formatter) -> Result<(), fmt::Error> {
    let mut items: Vec<String> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        items.push(format!("{}", expr));
    }
    write_list(&items, f, true)?;
    Ok(())
}

fn write_list<T: Display>(
    values: &Vec<T>,
    f: &mut Formatter,
    use_parens: bool,
) -> Result<(), fmt::Error> {
    if use_parens {
        write!(f, "(")?;
    }
    for (i, arg) in values.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    if use_parens {
        write!(f, ")")?;
    }
    Ok(())
}

fn clone_ast_expr_vec(vec: &Vec<ast::BExpression>) -> Vec<BExpression> {
    vec.iter().map(|x| {
        Box::new(Expression::from(x.as_ref()))
    }).collect::<Vec<BExpression>>()
}
