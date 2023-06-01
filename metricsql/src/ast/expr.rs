use crate::common::{
    format_num, write_list, AggregateModifier, BinModifier, GroupModifier, GroupModifierOp,
    JoinModifier, LabelFilter, LabelFilterExpr, LabelFilterOp, Operator, StringExpr, Value,
    ValueType, VectorMatchCardinality, NAME_LABEL,
};
use crate::functions::{AggregateFunction, BuiltinFunction, TransformFunction};
use enquote::enquote;
use lib::{fmt_duration_ms, hash_f64};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::btree_set::BTreeSet;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Neg, Range};
use std::str::FromStr;
use std::{fmt, iter, ops};
use crate::ast::expr_equals;

use crate::parser::{escape_ident, ParseError, ParseResult};
use crate::prelude::{BuiltinFunctionType, get_aggregate_arg_idx_for_optimization};

pub type BExpr = Box<Expr>;

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn cast(self) -> Expr;
}

/// NumberExpr represents number expression.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct NumberLiteral {
    /// value is the parsed number, i.e. `1.23`, `-234`, etc.
    pub value: f64,
}

impl NumberLiteral {
    pub fn new(v: f64) -> Self {
        NumberLiteral {
            value: v,
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Value for NumberLiteral {
    fn value_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl From<f64> for NumberLiteral {
    fn from(value: f64) -> Self {
        NumberLiteral::new(value)
    }
}

impl From<i64> for NumberLiteral {
    fn from(value: i64) -> Self {
        NumberLiteral::new(value as f64)
    }
}

impl From<usize> for NumberLiteral {
    fn from(value: usize) -> Self {
        NumberLiteral::new(value as f64)
    }
}

impl Into<f64> for NumberLiteral {
    fn into(self) -> f64 {
        self.value
    }
}

impl PartialEq<NumberLiteral> for NumberLiteral {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value || self.value.is_nan() && other.value.is_nan()
    }
}

impl ExpressionNode for NumberLiteral {
    fn cast(self) -> Expr {
        Expr::Number(self.clone())
    }
}

impl Display for NumberLiteral {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        format_num(f, self.value)
    }
}

impl Hash for NumberLiteral {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(state, self.value);
    }
}

impl Neg for NumberLiteral {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let value = -self.value;
        NumberLiteral {
            value,
        }
    }
}

impl Deref for NumberLiteral {
    type Target = f64;

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

    pub fn return_type(&self) -> ValueType {
        ValueType::Scalar
    }
}

impl Display for DurationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.requires_step {
            write!(f, "{}i", self.value)
        } else {
            fmt_duration_ms(f, self.value)
        }
    }
}

// todo: MetricExpr => Selector
/// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Debug, Clone, Hash, Eq, Serialize, Deserialize)]
pub struct MetricExpr {
    /// LabelFilters contains a list of label filters from curly braces.
    /// Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilter>,
    /// label_filter_expressions contains a list of label filter expressions from WITH clause.
    /// This is transformed into label_filters during compilation.
    pub(crate) label_filter_expressions: Vec<LabelFilterExpr>,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilter::new(LabelFilterOp::Equal, NAME_LABEL, name.into()).unwrap();
        MetricExpr {
            label_filters: vec![name_filter],
            label_filter_expressions: vec![],
        }
    }

    pub fn with_filters(filters: Vec<LabelFilter>) -> Self {
        MetricExpr {
            label_filters: filters,
            label_filter_expressions: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.label_filters.is_empty() && self.label_filter_expressions.is_empty()
    }

    pub fn is_resolved(&self) -> bool {
        self.label_filter_expressions.is_empty()
            || self
                .label_filter_expressions
                .iter()
                .all(|x| x.is_resolved())
    }

    pub fn has_non_empty_metric_group(&self) -> bool {
        if !self.label_filters.is_empty() {
            // we should have at least the name filter
            return self.label_filters[0].is_metric_name_filter();
        }
        if self.label_filter_expressions.is_empty() {
            return false;
        }
        self.label_filter_expressions[0].is_metric_name_filter()
    }

    pub fn is_only_metric_group(&self) -> bool {
        if !self.has_non_empty_metric_group() {
            return false;
        }
        self.label_filters.len() == 1
    }

    pub fn name(&self) -> Option<&str> {
        match self
            .label_filters
            .iter()
            .find(|filter| filter.is_name_label())
        {
            Some(f) => Some(&f.value),
            None => None,
        }
    }

    pub fn add_tag<S: Into<String>>(&mut self, name: S, value: &str) {
        let name_str = name.into();
        for label in self.label_filters.iter_mut() {
            if label.label == name_str {
                label.value.clear();
                label.value.push_str(value);
                return;
            }
        }
        self.label_filters.push(LabelFilter {
            op: LabelFilterOp::Equal,
            label: name_str,
            value: value.to_string(),
        });
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }

    pub fn is_expanded(&self) -> bool {
        self.label_filter_expressions.is_empty()
    }

    pub fn to_label_filters(&self) -> ParseResult<Vec<LabelFilter>> {
        if !self.is_expanded() {
            // todo: err
        }
        let mut items = Vec::with_capacity(self.label_filter_expressions.len());
        for filter in self.label_filter_expressions.iter() {
            items.push(filter.to_label_filter()?)
        }
        Ok(items)
    }

    pub fn is_empty_matchers(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        let mut is_empty = self.label_filters.iter().all(|x| x.is_empty_matcher());
        if !is_empty {
            is_empty = self
                .label_filter_expressions
                .iter()
                .all(|x| x.is_empty_matcher())
        }
        is_empty
    }

    /// find all the matchers whose name equals the specified name.
    pub fn find_matchers(&self, name: &str) -> Vec<&LabelFilter> {
        self.label_filters
            .iter()
            .filter(|m| m.label.eq_ignore_ascii_case(name))
            .collect()
    }

    pub fn sort_filters(&mut self) {
        self.label_filters.sort();
    }
}

impl Value for MetricExpr {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.is_empty() {
            write!(f, "{{}}")?;
            return Ok(());
        }

        let mut name_written = false;
        let mut lfs: &[LabelFilter] = &self.label_filters;
        let mut exprs: &[LabelFilterExpr] = &self.label_filter_expressions;

        if !lfs.is_empty() {
            let lf = &lfs[0];
            if lf.is_name_label() {
                write!(f, "{}", &lf.value)?;
                name_written = true;
                lfs = &lfs[1..];
            }
        }
        if !name_written && !exprs.is_empty() {
            let expr = &exprs[0];
            if expr.is_name_label() {
                write!(f, "{}", &expr.value)?;
                exprs = &exprs[1..];
            }
        }

        let total = lfs.len() + exprs.len();
        if total > 0 {
            write!(f, "{{")?;
        }
        if lfs.len() > 0 {
            write_list(lfs.iter(), f, false)?;
        }
        if !exprs.is_empty() {
            write_list(exprs.iter(), f, false)?;
        }
        if total > 0 {
            write!(f, "}}")?;
        }
        Ok(())
    }
}

impl Default for MetricExpr {
    fn default() -> Self {
        Self {
            label_filters: vec![],
            label_filter_expressions: vec![],
        }
    }
}

impl PartialEq<MetricExpr> for MetricExpr {
    fn eq(&self, other: &MetricExpr) -> bool {
        if self.label_filters.len() != other.label_filters.len() {
            return false;
        }
        if self.label_filter_expressions.len() != other.label_filter_expressions.len() {
            return false;
        }
        if !self.label_filters.is_empty() {
            let set = self
                .label_filters
                .iter()
                .map(|x| x.to_string())
                .collect::<BTreeSet<_>>();

            if !other
                .label_filters
                .iter()
                .all(|x| set.contains(&x.to_string()))
            {
                return false;
            }
        }
        if !self.label_filter_expressions.is_empty() {
            let set = self
                .label_filter_expressions
                .iter()
                .map(|x| x.to_string())
                .collect::<BTreeSet<_>>();

            if !other
                .label_filter_expressions
                .iter()
                .all(|x| set.contains(&x.to_string()))
            {
                return false;
            }
        }

        true
    }
}

impl ExpressionNode for MetricExpr {
    fn cast(self) -> Expr {
        Expr::MetricExpression(self)
    }
}

/// FuncExpr represents MetricsQL function such as `rate(...)`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionExpr {
    pub name: String,

    /// Args contains function args.
    pub args: Vec<Expr>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_idx_for_optimization: Option<usize>,

    /// If keep_metric_names is set to true, then the function should keep metric names.
    pub keep_metric_names: bool,

    pub is_scalar: bool,

    pub function_type: BuiltinFunctionType,
    pub return_type: ValueType,
}

impl FunctionExpr {
    pub fn new(name: &str, args: Vec<Expr>) -> ParseResult<Self> {
        let func_name = if name.is_empty() { "union" } else { name };
        let function = BuiltinFunction::new(func_name)?;
        let return_type = function.return_type(&args)?; // TODO
        let is_scalar = function.is_scalar();
        let arg_idx = function.get_arg_idx_for_optimization(args.len());

        Ok(Self {
            name: name.to_string(),
            args,
            arg_idx_for_optimization: arg_idx,
            keep_metric_names: false,
            is_scalar,
            function_type: function.get_type(),
            return_type,
        })
    }

    pub fn return_type(&self) -> ValueType {
        self.return_type
    }

    pub fn get_arg_for_optimization(&self) -> Option<&Expr> {
        match self.arg_idx_for_optimization {
            None => None,
            Some(idx) => Some(&self.args[idx]),
        }
    }

    pub fn default_rollup(arg: Expr) -> ParseResult<Self> {
        Self::from_single_arg("default_rollup", arg)
    }

    pub fn from_single_arg(name: &str, arg: Expr) -> ParseResult<Self> {
        let args = vec![arg];
        Self::new(name, args)
    }

    pub fn is_rollup(&self) -> bool {
        self.function_type == BuiltinFunctionType::Rollup
    }
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        write_list(&mut self.args.iter(), f, true)?;
        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

/// AggregationExpr represents aggregate function such as `sum(...) by (...)`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregationExpr {
    /// name is the aggregation function name.
    pub name: String,

    /// function args.
    pub args: Vec<Expr>,

    /// optional modifier such as `by (...)` or `without (...)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<AggregateModifier>,

    /// optional limit for the number of output time series.
    /// This is an MetricsQL extension.
    ///
    /// Example: `sum(...) by (...) limit 10` would return maximum 10 time series.
    pub limit: usize,

    pub keep_metric_names: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_idx_for_optimization: Option<usize>,
}

impl AggregationExpr {
    pub fn new(function: &AggregateFunction, args: Vec<Expr>) -> AggregationExpr {
        let arg_len = args.len();
        AggregationExpr {
            name: function.to_string(),
            args,
            modifier: None,
            limit: 0,
            keep_metric_names: false,
            arg_idx_for_optimization: get_aggregate_arg_idx_for_optimization(*function, arg_len),
        }
    }

    pub fn from_name(name: &str) -> ParseResult<Self> {
        let function = AggregateFunction::from_str(name)?;
        Ok(Self::new(&function, vec![]))
    }

    pub fn with_modifier(mut self, modifier: AggregateModifier) -> Self {
        self.modifier = Some(modifier);
        self
    }

    pub fn with_args(mut self, args: &[Expr]) -> Self {
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
            Expr::Function(fe) => {
                self.keep_metric_names = fe.keep_metric_names;
            }
            _ => self.keep_metric_names = false,
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }

    pub fn get_arg_for_optimization(&self) -> Option<&'_ Expr> {
        match self.arg_idx_for_optimization {
            None => None,
            Some(idx) => Some(&self.args[idx]),
        }
    }
}

impl Display for AggregationExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        let args_len = self.args.len();
        if args_len > 0 {
            write_list(&mut self.args.iter(), f, true)?;
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// RollupExpr represents an MetricsQL expression which contains at least `offset` or `[...]` part.
pub struct RollupExpr {
    /// The expression for the rollup. Usually it is MetricExpr, but may be arbitrary expr
    /// if subquery is used. https://prometheus.io/blog/2019/01/28/subquery-support/
    pub expr: BExpression,

    /// window contains optional window value from square brackets. Equivalent to `range` in
    /// prometheus terminology
    ///
    /// For example, `http_requests_total[5m]` will have Window value `5m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<DurationExpr>,

    /// step contains optional step value from square brackets. Equivalent to `resolution`
    /// in the prometheus docs
    ///
    /// For example, `foobar[1h:3m]` will have step value `3m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<DurationExpr>,

    /// offset contains optional value from `offset` part.
    ///
    /// For example, `foobar{baz="aa"} offset 5m` will have Offset value `5m`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<DurationExpr>,

    /// if set to true, then `foo[1h:]` would print the same instead of `foo[1h]`.
    pub inherit_step: bool,

    /// at contains an optional expression after `@` modifier.
    ///
    /// For example, `foo @ end()` or `bar[5m] @ 12345`
    /// See https://prometheus.io/docs/prometheus/latest/querying/basics/#modifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<BExpression>,
}

impl RollupExpr {
    pub fn new(expr: Expr) -> Self {
        RollupExpr {
            expr: Box::new(expr),
            window: None,
            offset: None,
            step: None,
            inherit_step: false,
            at: None,
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

    pub fn set_window(mut self, expr: DurationExpr) -> ParseResult<Self> {
        self.window = Some(expr);
        self.validate().map_err(|e| ParseError::General(e.into()))?;
        Ok(self)
    }

    pub fn set_expr(&mut self, expr: impl ExpressionNode) -> ParseResult<()> {
        self.expr = Box::new(expr.cast());
        self.validate().map_err(|e| ParseError::General(e.into()))
    }

    fn validate(&self) -> Result<(), String> {
        // range + subquery is not allowed (however this is syntactically invalid)
        // if self.window.is_some() && self.for_subquery() {
        //     return Err(
        //         "range and subquery are not allowed together in a rollup expression".to_string(),
        //     );
        // }
        Ok(())
    }

    pub fn return_type(&self) -> ValueType {
        // sub queries turn instant vectors into ranges
        return match (self.window.is_some(), self.for_subquery()) {
            (false, false) => ValueType::InstantVector,
            (false, true) => ValueType::RangeVector,
            (true, false) => ValueType::RangeVector,
            (true, true) => {
                ValueType::RangeVector
                // unreachable!("range and subquery are not allowed together in a rollup expression")
            }
        };
    }

    pub fn wraps_metric_expr(&self) -> bool {
        match *self.expr {
            Expr::MetricExpression(_) => true,
            _ => false,
        }
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let need_parens = match self.expr.as_ref() {
            Expr::Rollup(_) => true,
            Expr::BinaryOperator(_) => true,
            Expr::Aggregation(ae) => ae.modifier.is_some(),
            _ => false,
        };
        if need_parens {
            write!(f, "(")?;
        }
        write!(f, "{}", self.expr)?;
        if need_parens {
            write!(f, ")")?;
        }

        if self.window.is_some() || self.inherit_step || self.step.is_some() {
            write!(f, "[")?;
            if let Some(win) = &self.window {
                write!(f, "{}", win)?;
            }
            if let Some(step) = &self.step {
                write!(f, ":")?;
                write!(f, "{}", step)?;
            } else if self.inherit_step {
                write!(f, ":")?;
            }
            write!(f, "]")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " offset ")?;
            write!(f, "{}", offset)?;
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

/// BinaryOpExpr represents a binary operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryExpr {
    /// left contains left arg for the `left op right` expression.
    pub left: BExpression,

    /// right contains right arg for the `left op right` expression.
    pub right: BExpression,

    /// Op is the operation itself, i.e. `+`, `-`, `*`, etc.
    pub op: Operator,

    /// bool_modifier indicates whether `bool` modifier is present.
    /// For example, `foo > bool bar`.
    pub bool_modifier: bool,

    /// If keep_metric_names is set to true, then the operation should keep metric names.
    pub keep_metric_names: bool,

    /// group_modifier contains modifier such as "on" or "ignoring".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_modifier: Option<GroupModifier>,

    /// join_modifier contains modifier such as "group_left" or "group_right".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_modifier: Option<JoinModifier>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<BinModifier>,
}

impl BinaryExpr {
    pub fn new(op: Operator, lhs: Expr, rhs: Expr) -> Self {
        // operators can only have instant vectors or scalars
        // TODO
        BinaryExpr {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            join_modifier: None,
            group_modifier: None,
            bool_modifier: false,
            keep_metric_names: false,
            modifier: None,
        }
    }

    /// Unary minus. Substitute `-expr` with `0 - expr`
    pub fn new_unary_minus(expr: Expr) -> Self {
        BinaryExpr::new(Operator::Sub, Expr::from(0.0), expr)
    }

    pub fn get_group_modifier_or_default(&self) -> (GroupModifierOp, Cow<Vec<String>>) {
        match &self.group_modifier {
            None => (GroupModifierOp::Ignoring, Cow::Owned::<Vec<String>>(vec![])),
            Some(modifier) => (modifier.op, Cow::Borrowed(&modifier.labels)),
        }
    }

    pub fn is_matching_on(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_matching_on())
    }

    pub fn is_matching_labels_not_empty(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_matching_labels_not_empty())
    }

    pub fn return_bool(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.return_bool)
    }

    /// check if labels of card and matching are joint
    pub fn is_labels_joint(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.is_labels_joint())
    }

    /// intersect labels of card and matching
    pub fn intersect_labels(&self) -> Option<Vec<&String>> {
        self.modifier
            .as_ref()
            .and_then(|modifier| modifier.intersect_labels())
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

        if Expr::is_number(&self.right) || !Expr::is_scalar(&self.left) {
            return false;
        }
        true
    }

    pub fn return_type(&self) -> ValueType {
        let lhs_ret = self.left.return_type();
        let rhs_ret = self.right.return_type();

        match (lhs_ret, rhs_ret) {
            (ValueType::Scalar, ValueType::Scalar) => ValueType::Scalar,
            (ValueType::RangeVector, ValueType::RangeVector) => ValueType::RangeVector,
            (ValueType::InstantVector, ValueType::InstantVector) => ValueType::InstantVector,
            (ValueType::InstantVector, ValueType::Scalar) => ValueType::InstantVector,
            (ValueType::Scalar, ValueType::InstantVector) => ValueType::InstantVector,
            (ValueType::String, ValueType::String) => {
                if self.op.is_comparison() {
                    return ValueType::Scalar;
                }
                debug_assert!(
                    self.op == Operator::Add,
                    "Operator {} is not valid for (String, String)",
                    self.op
                );
                return ValueType::String;
            }
            _ => return ValueType::InstantVector,
        }
    }

    pub fn returns_bool(&self) -> bool {
        matches!(&self.modifier, Some(modifier) if modifier.return_bool)
    }

    pub fn vector_match_cardinality(&self) -> Option<&VectorMatchCardinality> {
        if let Some(modifier) = self.modifier.as_ref() {
            return Some(&modifier.card);
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
                        return Err(ParseError::SyntaxError(msg));
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

        if self.keep_metric_names {
            write!(f, " keep_metric_names")?;
        }
        Ok(())
    }
}

// TODO: ParensExpr => GroupExpr
/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParensExpr {
    pub expressions: Vec<Expr>,
}

impl ParensExpr {
    pub fn new(expressions: Vec<Expr>) -> Self {
        ParensExpr { expressions }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }

    pub fn return_type(&self) -> ValueType {
        if self.len() == 1 {
            return self.expressions[0].return_type();
        }

        // Treat as a function with empty name, i.e. union()
        TransformFunction::Union.return_type()
    }

    pub fn to_function(self) -> FunctionExpr {
        // Treat parensExpr as a function with empty name, i.e. union()
        // todo: how to avoid clone
        let name = "union";
        let func = BuiltinFunction::from_str(name).unwrap(); // if union is not defined, we have a fatal issue

        let arg_idx = func.get_arg_idx_for_optimization(self.len());
        FunctionExpr {
            name: name.to_string(),
            args: self.expressions,
            keep_metric_names: false,
            is_scalar: false,
            return_type: TransformFunction::Union.return_type(),
            function_type: func.get_type(),
            arg_idx_for_optimization: arg_idx,
        }
    }
}

impl Value for ParensExpr {
    fn value_type(&self) -> ValueType {
        if self.len() == 1 {
            return self.expressions[0].return_type();
        }

        // Treat as a function with empty name, i.e. union()
        TransformFunction::Union.return_type()
    }
}

impl Display for ParensExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_list(&mut self.expressions.iter(), f, true)?;
        Ok(())
    }
}

impl ExpressionNode for ParensExpr {
    fn cast(self) -> Expr {
        Expr::Parens(self)
    }
}

/// WithExpr represents `with (...)` extension from MetricsQL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithExpr {
    pub was: Vec<WithArgExpr>,
    pub expr: BExpr,
}

impl WithExpr {
    pub fn new(expr: Expr, was: Vec<WithArgExpr>) -> Self {
        WithExpr {
            expr: Box::new(expr),
            was,
        }
    }

    pub fn return_type(&self) -> ValueType {
        self.expr.return_type()
    }
}

impl Value for WithExpr {
    fn value_type(&self) -> ValueType {
        self.expr.value_type()
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
    fn cast(self) -> Expr {
        Expr::With(self)
    }
}

/// WithArgExpr represents a single entry from WITH expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithArgExpr {
    pub name: String,
    pub args: Vec<String>,
    pub expr: Expr,
    pub(crate) token_range: Range<usize>,
}

impl PartialEq for WithArgExpr {
    fn eq(&self, other: &Self) -> bool {
        let res = self.name == other.name &&
            self.args == other.args &&
            expr_equals(&self.expr, &other.expr);
        res
    }
}

impl WithArgExpr {
    pub fn new_function<S: Into<String>>(name: S, expr: Expr, args: Vec<String>) -> Self {
        WithArgExpr {
            name: name.into(),
            args,
            expr,
            token_range: Default::default(),
        }
    }

    pub fn new<S: Into<String>>(name: S, expr: Expr, args: Vec<String>) -> Self {
        WithArgExpr {
            name: name.into(),
            args,
            expr,
            token_range: Default::default(),
        }
    }

    pub fn new_number<S: Into<String>>(name: S, value: f64) -> Self {
        Self::new(name, Expr::from(value), vec![])
    }

    pub fn new_string<S: Into<String>>(name: S, value: String) -> Self {
        Self::new(name, Expr::from(value), vec![])
    }

    pub fn return_value(&self) -> ValueType {
        self.expr.return_type()
    }
}

impl Value for WithArgExpr {
    fn value_type(&self) -> ValueType {
        self.expr.value_type()
    }
}

impl Display for WithArgExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.name))?;
        write_list(self.args.iter(), f, !self.args.is_empty())?;
        write!(f, " = {}", self.expr)?;
        Ok(())
    }
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// A single scalar number.
    Number(NumberLiteral),

    Duration(DurationExpr),

    /// A single scalar string.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    StringLiteral(String),

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

    /// A grouped expression wrapped in parentheses
    Parens(ParensExpr),

    /// String expression parsed in the context of a `with` statement.
    ///
    /// Prometheus' docs claim strings aren't currently implemented, but they're
    /// valid as function arguments.
    StringExpr(StringExpr),

    /// A MetricsQL specific WITH statement node. Transformed at parse time to one
    /// of the other variants
    With(WithExpr),
}

pub type BExpression = Box<Expr>;

impl Expr {
    pub fn is_scalar(expr: &Expr) -> bool {
        match expr {
            Expr::Duration(_) | Expr::Number(_) => true,
            Expr::Function(f) => f.is_scalar,
            _ => false,
        }
    }

    pub fn is_number(expr: &Expr) -> bool {
        matches!(expr, Expr::Number(_))
    }

    pub fn is_string(expr: &Expr) -> bool {
        matches!(expr, Expr::StringLiteral(_))
    }

    pub fn is_primitive(expr: &Expr) -> bool {
        matches!(expr, Expr::Number(_) | Expr::StringLiteral(_))
    }

    pub fn is_duration(expr: &Expr) -> bool {
        matches!(expr, Expr::Duration(_))
    }

    pub fn vectors(&self) -> Box<dyn Iterator<Item = &LabelFilter> + '_> {
        match self {
            Self::MetricExpression(v) => Box::new(v.label_filters.iter()),
            Self::Rollup(re) => Box::new(re.expr.vectors().chain(if let Some(at) = &re.at {
                at.vectors()
            } else {
                Box::new(iter::empty())
            })),
            Self::BinaryOperator(be) => Box::new(be.left.vectors().chain(be.right.vectors())),
            Self::Aggregation(ae) => Box::new(ae.args.iter().flat_map(|node| node.vectors())),
            Self::Function(fe) => Box::new(fe.args.iter().flat_map(|node| node.vectors())),
            Self::Parens(pe) => Box::new(pe.expressions.iter().flat_map(|node| node.vectors())),
            Self::Number(_)
            | Self::Duration(_)
            | Self::StringLiteral(_)
            | Self::StringExpr(_)
            | Self::With(_) => Box::new(iter::empty()),
        }
    }

    /**
    Return an iterator of series names present in this node.
    ```
    let query = r#"
        sum(1 - something_used{env="production"} / something_total) by (instance)
        and ignoring (instance)
        sum(rate(some_queries{instance=~"localhost\\d+"} [5m])) > 100
    "#;
    let ast = promql::parse(query).expect("valid query");
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
            if x.label == NAME_LABEL {
                x.value.clone()
                // String::from_utf8(x.value.clone())
                //     .expect("series names should always be valid utf8")
            } else {
                x.label.clone()
            }
        })
    }

    pub fn contains_subquery(&self) -> bool {
        use Expr::*;
        match self {
            Function(fe) => fe.args.iter().any(|e| e.contains_subquery()),
            BinaryOperator(bo) => bo.left.contains_subquery() || bo.right.contains_subquery(),
            Aggregation(aggr) => aggr.args.iter().any(|e| e.contains_subquery()),
            Rollup(re) => re.for_subquery(),
            _ => false,
        }
    }

    pub fn return_type(&self) -> ValueType {
        match self {
            Expr::Number(_) => ValueType::Scalar,
            Expr::Duration(dur) => dur.return_type(),
            Expr::StringLiteral(_) | Expr::StringExpr(_) => ValueType::String,
            Expr::Function(fe) => fe.return_type(),
            Expr::Aggregation(ae) => ae.return_type(),
            Expr::BinaryOperator(be) => be.return_type(),
            Expr::Rollup(re) => re.return_type(),
            Expr::Parens(me) => me.return_type(),
            Expr::MetricExpression(me) => me.return_type(),
            Expr::With(w) => w.return_type(),
        }
    }

    pub fn variant_name(&self) -> &str {
        match self {
            Expr::Number(_) => "Scalar",
            Expr::Duration(_) => "Scalar",
            Expr::StringLiteral(_) | Expr::StringExpr(_) => "String",
            Expr::Function(_) => "Function",
            Expr::Aggregation(_) => "Aggregation",
            Expr::BinaryOperator(_) => "BinaryOperator",
            Expr::Rollup(_) => "Rollup",
            Expr::Parens(_) => "Parens",
            Expr::MetricExpression(_) => "VectorSelector",
            Expr::With(_) => "With",
        }
    }

    pub fn cast(self) -> Expr {
        // this code seems suspicious
        match self {
            Expr::Aggregation(a) => Expr::Aggregation(a),
            Expr::BinaryOperator(b) => Expr::BinaryOperator(b),
            Expr::Duration(d) => Expr::Duration(d),
            Expr::Function(f) => Expr::Function(f),
            Expr::Number(n) => Expr::Number(n),
            Expr::MetricExpression(m) => Expr::MetricExpression(m),
            Expr::Parens(m) => Expr::Parens(m),
            Expr::Rollup(r) => Expr::Rollup(r),
            Expr::StringLiteral(s) => Expr::StringLiteral(s),
            Expr::StringExpr(s) => Expr::StringExpr(s),
            Expr::With(w) => Expr::With(w),
        }
    }

    pub fn is_metric_expression(&self) -> bool {
        match self {
            Expr::MetricExpression(_) => true,
            _ => false,
        }
    }

    pub fn is_binary_op(&self) -> bool {
        match self {
            Expr::BinaryOperator(_) => true,
            _ => false,
        }
    }

    /// returns a scalar expression
    pub fn scalar(value: f64) -> Expr {
        Expr::from(value)
    }

    /// returns a string literal expression
    pub fn string_literal(value: &str) -> Expr {
        Expr::from(value)
    }

    /// Return `self == other`
    pub fn eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Eql, other)
    }

    /// Return `self != other`
    pub fn not_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::NotEq, other)
    }

    /// Return `self > other`
    pub fn gt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gt, other)
    }

    /// Return `self >= other`
    pub fn gt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Gte, other)
    }

    /// Return `self < other`
    pub fn lt(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lt, other)
    }

    /// Return `self <= other`
    pub fn lt_eq(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Lte, other)
    }

    /// Return `self AND other`
    pub fn and(self, other: Expr) -> Expr {
        binary_expr(self, Operator::And, other)
    }

    /// Return `self OR other`
    pub fn or(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Or, other)
    }

    /// Calculate the modulus of two expressions.
    /// Return `self % other`
    pub fn modulus(self, other: Expr) -> Expr {
        binary_expr(self, Operator::Mod, other)
    }

    pub fn call(func: &str, args: Vec<Expr>) -> ParseResult<Expr> {
        let expr = FunctionExpr::new(func, args)?;
        Ok(Expr::Function(expr))
    }

    pub fn new_binary_expr(
        lhs: Expr,
        op: Operator,
        modifier: Option<BinModifier>,
        rhs: Expr,
    ) -> Result<Expr, String> {
        let ex = BinaryExpr {
            left: Box::new(lhs),
            right: Box::new(rhs),
            op,
            bool_modifier: false,
            keep_metric_names: false,
            group_modifier: None,
            modifier,
            join_modifier: None,
        };
        Ok(Expr::BinaryOperator(ex))
    }

}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Expr::Aggregation(a) => write!(f, "{}", a)?,
            Expr::BinaryOperator(be) => write!(f, "{}", be)?,
            Expr::Duration(d) => write!(f, "{}", d)?,
            Expr::Function(func) => write!(f, "{}", func)?,
            Expr::Number(n) => write!(f, "{}", n)?,
            Expr::MetricExpression(me) => write!(f, "{}", me)?,
            Expr::Parens(p) => write!(f, "({})", p)?,
            Expr::Rollup(re) => write!(f, "{}", re)?,
            Expr::StringLiteral(s) => write!(f, "{}", enquote('"', s))?,
            Expr::StringExpr(s) => write!(f, "{}", s)?,
            Expr::With(w) => write!(f, "{}", w)?,
        }
        Ok(())
    }
}

impl Value for Expr {
    fn value_type(&self) -> ValueType {
        match self {
            Expr::Aggregation(a) => a.return_type(),
            Expr::BinaryOperator(be) => be.return_type(),
            Expr::Duration(d) => d.return_type(),
            Expr::Function(func) => func.return_type(),
            Expr::Number(n) => n.return_type(),
            Expr::MetricExpression(me) => me.value_type(),
            Expr::Parens(p) => p.return_type(),
            Expr::Rollup(re) => re.return_type(),
            Expr::StringLiteral(_) => ValueType::String,
            Expr::StringExpr(_) => ValueType::String,
            Expr::With(w) => w.return_type(),
        }
    }
}

// crate private
impl Default for Expr {
    fn default() -> Self {
        Expr::from(1.0)
    }
}

impl From<f64> for Expr {
    fn from(v: f64) -> Self {
        Expr::Number(NumberLiteral::new(v))
    }
}

impl From<i64> for Expr {
    fn from(v: i64) -> Self {
        Self::from(v as f64)
    }
}

impl From<String> for Expr {
    fn from(s: String) -> Self {
        Expr::StringLiteral(s)
    }
}

impl From<&str> for Expr {
    fn from(s: &str) -> Self {
        Expr::StringLiteral(s.to_string())
    }
}

pub(crate) fn binary_expr(left: Expr, op: Operator, right: Expr) -> Expr {
    let mut expr = BinaryExpr::new(op, left, right);
    expr.bool_modifier = op.is_comparison();
    Expr::BinaryOperator(expr)
}

impl ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Add, rhs)
    }
}

impl ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Sub, rhs)
    }
}

impl ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mul, rhs)
    }
}

impl ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Div, rhs)
    }
}

impl ops::Rem for Expr {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Mod, rhs)
    }
}

impl ops::BitAnd for Expr {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        binary_expr(self, Operator::And, rhs)
    }
}

impl ops::BitOr for Expr {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Or, rhs)
    }
}

impl ops::BitXor for Expr {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        binary_expr(self, Operator::Pow, rhs)
    }
}

fn intersection(labels_a: &Vec<String>, labels_b: &Vec<String>) -> Vec<String> {
    if labels_a.is_empty() || labels_b.is_empty() {
        return vec![];
    }
    let unique_a: HashSet<String> = labels_a.clone().into_iter().collect();
    let unique_b: HashSet<String> = labels_b.clone().into_iter().collect();
    unique_a
        .intersection(&unique_b)
        .map(|i| i.clone())
        .collect::<Vec<_>>()
}
