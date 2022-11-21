use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{
    Expression, ExpressionNode, LabelFilter,
    LabelFilterExpr, LabelFilterOp, ReturnValue
};
use crate::ast::expression_kind::ExpressionKind;
use crate::lexer::TextSpan;
use crate::utils::escape_ident;
use serde::{Serialize, Deserialize};

// todo: MetricExpr => Selector
/// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct MetricExpr {
    /// LabelFilters contains a list of label filters from curly braces.
    /// Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilter>,

    /// label_filters must be expanded to LabelFilters by expand_with_expr.
    #[serde(skip)]
    pub(crate) label_filter_exprs: Vec<LabelFilterExpr>,

    pub span: TextSpan,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilter::new(LabelFilterOp::Equal, "__name__", name).unwrap();
        MetricExpr {
            label_filters: vec![name_filter],
            label_filter_exprs: vec![],
            span: TextSpan::default(),
        }
    }

    pub fn with_filters(filters: Vec<LabelFilter>) -> Self {
        MetricExpr {
            label_filters: filters,
            label_filter_exprs: vec![],
            span: Default::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.label_filters.len() == 0 && self.label_filter_exprs.len() == 0
    }

    pub fn is_expanded(&self) -> bool {
        !self.label_filters.is_empty()
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

    pub fn name(&mut self) -> Option<&str> {
        match self.label_filters.iter().find(|filter| filter.label == "__name__" ) {
            Some(f) => Some(&f.value),
            None => None
        }
    }

    pub fn add_tag<S: Into<String>>(&mut self, name: S, value: S) {
        let name_str = name.into();
        for label in self.label_filters.iter_mut() {
            if label.label == name_str {
                label.value = value.into();
                return;
            }
        }
        self.label_filters.push(LabelFilter {
            op: LabelFilterOp::Equal,
            label: name_str,
            value: value.into()
        });
    }

    pub fn return_value(&self) -> ReturnValue {
        ReturnValue::InstantVector
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut lfs: &[LabelFilter] = &self.label_filters;
        let mut name_written = false;
        if !lfs.is_empty() {
            let lf = &lfs[0];
            if lf.label == "__name__" && !lf.is_negative() && !lf.is_regexp() {
                write!(f, "{}", escape_ident(&lf.value))?;
                lfs = &lfs[1..];
                name_written = true;
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
            if !name_written {
                write!(f, "{{}}")?;
            }
        }
        Ok(())
    }
}

impl Default for MetricExpr {
    fn default() -> Self {
        Self {
            label_filters: vec![],
            label_filter_exprs: vec![],
            span: TextSpan::default()
        }
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