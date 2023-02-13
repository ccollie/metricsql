use std::fmt;
use std::fmt::{Display, Formatter};
use crate::ast::{Expression, ExpressionNode, LabelFilter, LabelFilterOp, ReturnType};
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};
use crate::ast::label_filter_expr::LabelFilterExpr;
use crate::ast::segmented_string::SegmentedString;
use crate::prelude::ParseResult;


// todo: MetricExpr => Selector
/// MetricExpr represents MetricsQL metric with optional filters, i.e. `foo{...}`.
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct MetricExpr {
    /// LabelFilters contains a list of label filters from curly braces.
    /// Filter or metric name must be the first if present.
    pub label_filters: Vec<LabelFilterExpr>,
    pub span: TextSpan,
}

impl MetricExpr {
    pub fn new<S: Into<String>>(name: S) -> MetricExpr {
        let name_filter = LabelFilterExpr::new(LabelFilterOp::Equal, "__name__", name).unwrap();
        MetricExpr {
            label_filters: vec![name_filter],
            span: TextSpan::default(),
        }
    }

    pub fn with_filters(filters: Vec<LabelFilterExpr>) -> Self {
        MetricExpr {
            label_filters: filters,
            span: Default::default(),
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
            Some(f) => Some(&f.value.to_string()),
            None => None
        }
    }

    pub fn add_tag<S: Into<String>>(&mut self, name: S, value: S) {
        let name_str = name.into();
        for label in self.label_filters.iter_mut() {
            if label.label == name_str {
                label.value.set_from_string(value);
                return;
            }
        }
        self.label_filters.push(LabelFilterExpr {
            op: LabelFilterOp::Equal,
            label: name_str,
            value: SegmentedString::from(value.into()),
        });
    }

    pub fn return_type(&self) -> ReturnType {
        ReturnType::InstantVector
    }

    pub fn is_expanded(&self) -> bool {
        self.label_filters.len() == 0 ||
        self.label_filters.iter().all(|x| x.is_resolved())
    }

    pub fn to_label_filters(&self) -> ParseResult<Vec<LabelFilter>> {
        if !self.is_expanded() {
            // todo: err
        }
        let mut items = Vec::with_capacity(self.label_filters.len());
        for filter in self.label_filters.iter() {
            items.push(filter.to_label_filter()?)
        }
        Ok(items)
    }
}

impl Display for MetricExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut lfs: &[LabelFilterExpr] = &self.label_filters;
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

impl Default for MetricExpr {
    fn default() -> Self {
        Self {
            label_filters: vec![],
            span: TextSpan::default(),
        }
    }
}

impl ExpressionNode for MetricExpr {
    fn cast(self) -> Expression {
        Expression::MetricExpression(self)
    }
}