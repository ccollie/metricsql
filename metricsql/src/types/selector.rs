// (C) Copyright 2019-2020 Hewlett Packard Enterprise Development LP

use std::collections::HashSet;
use std::fmt;
use std::iter::FromIterator;
use crate::types::LabelFilter;

use super::misc::{PromDuration, Span, Subquery};
use super::return_value::{LabelSetOp, ReturnKind, ReturnValue};

/// A Selector that retrieves time series data from Prometheus
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Selector {
    pub metric: Option<String>,
    pub labels: Vec<LabelFilter>,
    pub range: Option<PromDuration>,
    pub offset: Option<PromDuration>,
    pub subquery: Option<Subquery>,

    pub span: Option<Span>
}

impl Selector {
    pub fn new() -> Self {
        Selector {
            metric: None,
            labels: vec![],
            range: None,
            offset: None,
            subquery: None,
            span: None
        }
    }

    /// Sets or replaces this Selector's metric
    pub fn metric<S: Into<String>>(mut self, metric: S) -> Self {
        self.metric = Some(metric.into());
        self
    }

    /// Clears this Selector's metric
    pub fn clear_metric(mut self) -> Self {
        self.metric = None;
        self
    }

    /// Adds a label to this Selector
    pub fn label(mut self, label: LabelFilter) -> Self {
        self.labels.push(label);
        self
    }

    /// Replaces this Selector's labels with the given set
    pub fn labels(mut self, labels: Vec<LabelFilter>) -> Self {
        self.labels = labels;
        self
    }

    /// Clears this Selector's set of labels
    pub fn clear_labels(mut self) -> Self {
        self.labels.clear();
        self
    }

    /// Sets or replaces this Selector's range
    pub fn range(mut self, range: PromDuration) -> Self {
        self.range = Some(range);
        self
    }

    /// Clears this Selector's range
    pub fn clear_range(mut self) -> Self {
        self.range = None;
        self
    }

    /// Sets or replaces this Selector's offset
    pub fn offset(mut self, offset: PromDuration) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn clear_offset(mut self) -> Self {
        self.offset = None;
        self
    }

    pub fn subquery(mut self, subquery: Subquery) -> Self {
        self.subquery = Some(subquery);
        self
    }

    pub fn clear_subquery(mut self) -> Self {
        self.subquery = None;
        self
    }

    pub fn span<S: Into<Span>>(mut self, span: S) -> Self {
        self.span = Some(span.into());
        self
    }

    pub fn return_value(&self) -> ReturnValue {
        let kind = match (self.range.is_some(), self.subquery.is_some()) {
            (false, false) => ReturnKind::InstantVector,
            (false, true) => ReturnKind::RangeVector,
            (true, false) => ReturnKind::RangeVector,

            // range + subquery is not allowed (however this is syntactically invalid)
            (true, true) => ReturnKind::unknown(
                "range and subquery are not allowed together",
                self.clone().wrap()
            )
        };

        // if a label is selected, we can infer that it must be present on the
        // output vector
        let mut label_ops = Vec::new();
        if !self.labels.is_empty() {
            label_ops.push(LabelSetOp::append(
                self.clone().wrap(),
                self.span,
                HashSet::from_iter(self.labels.iter().cloned().map(|l| l.key))
            ));
        }

        ReturnValue { kind, label_ops }
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(metric) = &self.metric {
            write!(f, "{}", metric)?;
        }

        if !self.labels.is_empty() {
            write!(f, "{{")?;
            for (i, label) in self.labels.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }

                write!(f, "{}", label)?;
            }

            write!(f, "}}")?;
        };

        if let Some(range) = &self.range {
            write!(f, "[{}]", range)?;
        }

        if let Some(offset) = &self.offset {
            write!(f, " offset {}", offset)?;
        }

        if let Some(subquery) = &self.subquery {
            write!(f, "{}", subquery)?;
        }

        Ok(())
    }
}