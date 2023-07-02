use std::collections::BTreeSet;
use std::fmt;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::common::{LabelFilter, LabelFilterExpr, NAME_LABEL, StringExpr, Value, ValueType, write_list};
use crate::parser::ParseResult;

/// InterpolatedSelector represents a MetricsQL metric in the context of a WITH expression.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
pub struct InterpolatedSelector {
    /// a list of label filter expressions from WITH clause.
    /// This is transformed into label_filters during compilation.
    pub(crate) matchers: Vec<LabelFilterExpr>,
}

impl InterpolatedSelector {
    pub fn new<S: Into<String>>(name: S) -> InterpolatedSelector {
        let name_filter = LabelFilterExpr::equal(NAME_LABEL, StringExpr::new(name)).unwrap();
        InterpolatedSelector {
            matchers: vec![name_filter],
        }
    }

    pub fn with_filters(filters: Vec<LabelFilterExpr>) -> Self {
        InterpolatedSelector {
            matchers: filters,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }

    pub fn is_resolved(&self) -> bool {
        self.matchers.is_empty()
            || self
            .matchers
            .iter()
            .all(|x| x.is_resolved())
    }

    pub fn has_non_empty_metric_group(&self) -> bool {
        if self.matchers.is_empty() {
            return false;
        }
        self.matchers[0].is_metric_name_filter()
    }

    pub fn is_only_metric_group(&self) -> bool {
        if !self.has_non_empty_metric_group() {
            return false;
        }
        self.matchers.len() == 1
    }

    pub fn name(&self) -> Option<String> {
        match self
            .matchers
            .iter()
            .find(|filter| filter.is_name_label())
        {
            Some(f) => Some(f.value.to_string()),
            None => None,
        }
    }

    pub fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }


    pub fn to_label_filters(&self) -> ParseResult<Vec<LabelFilter>> {
        if !self.is_resolved() {
            // todo: err
        }
        let mut items = Vec::with_capacity(self.matchers.len());
        for filter in self.matchers.iter() {
            items.push(filter.to_label_filter()?)
        }
        Ok(items)
    }

    pub fn is_empty_matchers(&self) -> bool {
        self.matchers.is_empty() ||
            self
                .matchers
                .iter()
                .all(|x| x.is_empty_matcher())
    }

    /// find all the matchers whose name equals the specified name.
    pub fn find_matchers(&self, name: &str) -> Vec<&LabelFilterExpr> {
        self.matchers
            .iter()
            .filter(|m| m.label.eq_ignore_ascii_case(name))
            .collect()
    }
}

impl Value for InterpolatedSelector {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Display for InterpolatedSelector {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.is_empty() {
            write!(f, "{{}}")?;
            return Ok(());
        }

        let mut exprs: &[LabelFilterExpr] = &self.matchers;

        if !exprs.is_empty() {
            let lf = &exprs[0];
            if lf.is_name_label() {
                write!(f, "{}", &lf.value)?;
                exprs = &exprs[1..];
            }
        }

        if !exprs.is_empty() {
            write!(f, "{{")?;
            write_list(exprs.iter(), f, false)?;
            write!(f, "}}")?;
        }

        Ok(())
    }
}

impl Default for InterpolatedSelector {
    fn default() -> Self {
        Self {
            matchers: vec![],
        }
    }
}

impl PartialEq<InterpolatedSelector> for InterpolatedSelector {
    fn eq(&self, other: &InterpolatedSelector) -> bool {
        if self.matchers.len() != other.matchers.len() {
            return false;
        }
        let mut hasher: Xxh3 = Xxh3::new();

        if !self.matchers.is_empty() {
            let mut set: BTreeSet<u64> = BTreeSet::new();
            for filter in &self.matchers {
                hasher.reset();
                filter.update_hash(&mut hasher);
                set.insert(hasher.digest());
            }

            for filter in &other.matchers {
                hasher.reset();
                filter.update_hash(&mut hasher);
                let hash = hasher.digest();
                if !set.contains(&hash) {
                    return false
                }
            }
        }

        true
    }
}
