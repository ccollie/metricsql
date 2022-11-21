use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{BExpression, Expression, ExpressionNode, ReturnValue};
use crate::ast::duration::DurationExpr;
use crate::ast::expression_kind::ExpressionKind;
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};

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

    pub span: TextSpan,
}

impl RollupExpr {
    pub fn new(expr: Expression) -> Self {
        let span = expr.span();
        RollupExpr {
            expr: Box::new(expr),
            window: None,
            offset: None,
            step: None,
            inherit_step: false,
            at: None,
            span,
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
                "range and subquery are not allowed together in a rollup expression",
                self.clone().cast()
            )
        };

        kind
    }

    pub fn wraps_metric_expr(&self) -> bool {
        match *self.expr {
            Expression::MetricExpression(_) => true,
            _ => false
        }
    }
}

impl Display for RollupExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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
            if let Some(win) = &self.window {
                write!(f, "{}", win)?;
            }
            if let Some(step) = &self.step {
                write!(f, ":{}", step)?;
            } else if self.inherit_step {
                write!(f, ":")?;
            }
            write!(f, "]")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " offset {}", offset)?;
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

impl ExpressionNode for RollupExpr {
    fn cast(self) -> Expression {
        Expression::Rollup(self)
    }

    fn kind(&self) -> ExpressionKind {
        ExpressionKind::Rollup
    }
}