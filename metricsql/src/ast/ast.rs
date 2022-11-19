use std::{fmt, iter};
use std::fmt::{Display, Formatter};
use std::hash::{Hash};
use std::string::{String};
use crate::ast::{AggrFuncExpr, BinaryOpExpr};

use crate::ast::duration::DurationExpr;
use crate::ast::expression_kind::ExpressionKind;
use crate::ast::function::FuncExpr;
use crate::ast::label_filter::{LabelFilter};
use crate::ast::misc::{write_expression_list};
use crate::ast::number::NumberExpr;
use crate::ast::return_type::ReturnValue;
use crate::ast::rollup::RollupExpr;
use crate::ast::selector::MetricExpr;
use crate::ast::string::StringExpr;
use crate::ast::with::WithExpr;
use crate::functions::{
   TransformFunction
};
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};

/// Expression Trait. Useful for cases where match is not ergonomic
pub trait ExpressionNode {
    fn cast(self) -> Expression;
    fn kind(&self) -> ExpressionKind;
}

/// A root expression node.
///
/// These are all valid root expression ast.
#[derive(Clone, Debug, Hash, Serialize, Deserialize)]
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

    pub fn span(&self) -> TextSpan {
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
        Expression::Parens(ParensExpr::new(list, TextSpan::default()))
    }
}

impl From<Vec<Expression>> for Expression {
    fn from(list: Vec<Expression>) -> Self {
        let items = list.into_iter().map(|x| Box::new(x)).collect::<Vec<BExpression>>();
        Expression::Parens(ParensExpr::new(items, TextSpan::default()))
    }
}

// todo: integrate checks from
// https://github.com/prometheus/prometheus/blob/fa6e05903fd3ce52e374a6e1bf4eb98c9f1f45a7/promql/parser/parse.go#L436


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

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Parens
    }
}