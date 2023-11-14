use std::fmt;
use std::fmt::{Display, Formatter};

use ahash::AHashSet;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::ast::{LabelFilterExpr, Prettier, StringExpr};
use crate::common::{Value, ValueType};
use crate::label::{LabelFilter, NAME_LABEL};
use crate::parser::ParseResult;

/// InterpolatedSelector represents a Vector Selector in the context of a WITH expression.
#[derive(Debug, Default, Clone, Eq, Serialize, Deserialize)]
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
        InterpolatedSelector { matchers: filters }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }

    pub fn is_resolved(&self) -> bool {
        self.matchers.is_empty() || self.matchers.iter().all(|x| x.is_resolved())
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
        self.matchers
            .iter()
            .find(|filter| filter.is_name_label())
            .map(|f| f.value.to_string())
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
        self.matchers.is_empty() || self.matchers.iter().all(|x| x.is_empty_matcher())
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
            for (i, arg) in exprs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, "}}")?;
        }

        Ok(())
    }
}

impl Prettier for InterpolatedSelector {
    fn needs_split(&self, _max: usize) -> bool {
        false
    }
}

impl PartialEq<InterpolatedSelector> for InterpolatedSelector {
    fn eq(&self, other: &InterpolatedSelector) -> bool {
        if self.matchers.len() != other.matchers.len() {
            return false;
        }
        let mut hasher: Xxh3 = Xxh3::new();

        if !self.matchers.is_empty() {
            let mut set: AHashSet<u64> = AHashSet::new();
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
                    return false;
                }
            }
        }

        true
    }
}
