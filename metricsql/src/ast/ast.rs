use std::{fmt, iter};
use std::fmt::{Display, Formatter};
use std::hash::{Hash};
use std::str::FromStr;
use std::string::{String, ToString};
use text_size::{TextRange};
use crate::ast::BinaryOpExpr;

use crate::ast::duration::DurationExpr;
use crate::ast::expression_kind::ExpressionKind;
use crate::ast::function::FuncExpr;
use crate::ast::label_filter::{LabelFilter};
use crate::ast::misc::{write_expression_list, write_labels};
use crate::ast::number::NumberExpr;
use crate::ast::return_type::ReturnValue;
use crate::ast::rollup::RollupExpr;
use crate::ast::selector::MetricExpr;
use crate::ast::string::StringExpr;
use crate::ast::with::WithExpr;
use crate::functions::{
   AggregateFunction,
   get_aggregate_arg_idx_for_optimization,
   TransformFunction
};
use crate::parser::{ParseError};

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

    pub fn is_metric_expression(&self) -> bool {
        match self {
            Expression::MetricExpression(_) => true,
            _ => false
        }
    }

    pub fn span(&self) -> TextRange {
        match self {
            Expression::Duration(de) => de.span,
            Expression::Number(num) => num.span,
            Expression::String(se) => se.span,
            Expression::Parens(pe) => pe.span,
            Expression::Function(fe) => fe.span,
            Expression::Aggregation(ae) => ae.span,
            Expression::BinaryOperator(be) => be.span,
            Expression::With(we) => we.span,
            Expression::Rollup(re) => re.span,
            Expression::MetricExpression(me) => me.span
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

// todo: integrate checks from
// https://github.com/prometheus/prometheus/blob/fa6e05903fd3ce52e374a6e1bf4eb98c9f1f45a7/promql/parser/parse.go#L436


#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum AggregateModifierOp {
    #[default]
    By,
    Without,
}

impl Display for AggregateModifierOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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
    pub span: Option<TextRange>,
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
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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

    pub span: TextRange,
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
            span: TextRange::default(),
        }
    }

    pub fn from_name(name: &str) -> Result<Self, ParseError> {
        let function = AggregateFunction::from_str(name)?;
        Ok(Self::new(&function))
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
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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


/// Expression(s) explicitly grouped in parens
#[derive(Default, Debug, Clone, Hash)]
pub struct ParensExpr {
    pub expressions: Vec<BExpression>,
    pub span: TextRange,
}

impl ParensExpr {
    pub fn new(expressions: Vec<BExpression>) -> Self {
        ParensExpr {
            expressions,
            span: TextRange::default(),
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

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Parens
    }
}