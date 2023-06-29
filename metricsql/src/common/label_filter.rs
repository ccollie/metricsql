use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hasher;
use std::ops::Deref;

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::parser::{compile_regexp, escape_ident, is_empty_regex, ParseError, quote};

pub const NAME_LABEL: &str = "__name__";

pub type LabelName = String;

pub type LabelValue = String;

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Copy, Hash, Serialize, Deserialize)]
pub enum LabelFilterOp {
    #[default]
    Equal,
    NotEqual,
    RegexEqual,
    RegexNotEqual,
}

impl LabelFilterOp {
    pub fn is_negative(&self) -> bool {
        matches!(self, LabelFilterOp::NotEqual | LabelFilterOp::RegexNotEqual)
    }

    pub fn is_regex(&self) -> bool {
        matches!(
            self,
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            LabelFilterOp::Equal => "=",
            LabelFilterOp::NotEqual => "!=",
            LabelFilterOp::RegexEqual => "=~",
            LabelFilterOp::RegexNotEqual => "!~",
        }
    }
}

impl TryFrom<&str> for LabelFilterOp {
    type Error = ParseError;

    fn try_from(op: &str) -> Result<Self, Self::Error> {
        match op {
            "=" => Ok(LabelFilterOp::Equal),
            "!=" => Ok(LabelFilterOp::NotEqual),
            "=~" => Ok(LabelFilterOp::RegexEqual),
            "!~" => Ok(LabelFilterOp::RegexNotEqual),
            _ => Err(ParseError::General(format!(
                "Unexpected match op literal: {}",
                op
            ))),
        }
    }
}

impl fmt::Display for LabelFilterOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// LabelFilter represents MetricsQL label filter like `foo="bar"`.
#[derive(Default, Debug, Clone, Eq, Hash, Serialize, Deserialize)]
pub struct LabelFilter {
    pub op: LabelFilterOp,

    /// Label contains label name for the filter.
    pub label: String,

    /// Value contains unquoted value for the filter.
    pub value: String,
}

impl LabelFilter {
    pub fn new<N, V>(match_op: LabelFilterOp, label: N, value: V) -> Result<Self, ParseError>
    where
        N: Into<LabelName>,
        V: Into<LabelValue>,
    {
        let label = label.into();
        let value = value.into();

        assert!(!label.is_empty());

        if match_op == LabelFilterOp::RegexEqual || match_op == LabelFilterOp::RegexNotEqual {
            let re_anchored = format!("^(?:{})$", value);
            if compile_regexp(&re_anchored).is_err() {
                return Err(ParseError::InvalidRegex(value));
            }
        }

        Ok(Self {
            label,
            op: match_op,
            value,
        })
    }

    pub fn equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::Equal, key, value)
    }

    pub fn not_equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::NotEqual, key, value)
    }

    pub fn regex_equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::RegexEqual, key, value)
    }

    pub fn regex_notequal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter, ParseError> {
        LabelFilter::new(LabelFilterOp::RegexNotEqual, key, value)
    }

    /// is_regexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op.is_regex()
    }

    /// is_negative represents whether the filter is negative, i.e. '!=' or '!~'.
    pub fn is_negative(&self) -> bool {
        self.op.is_negative()
    }

    pub fn is_metric_name_filter(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    pub fn is_name_label(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    /// Vector selectors must either specify a name or at least one label
    /// matcher that does not match the empty string.
    ///
    /// The following expression is illegal:
    /// {job=~".*"} # Bad!
    pub fn is_empty_matcher(&self) -> bool {
        use LabelFilterOp::*;
        match self.op {
            Equal => self.value.is_empty(),
            NotEqual => !self.value.is_empty(),
            RegexEqual => {
                let str = self.value.to_string();
                is_empty_regex(&str)
            }
            RegexNotEqual => {
                let str = self.value.to_string();
                is_empty_regex(&str)
            }
        }
    }

    pub fn as_string(&self) -> String {
        format!(
            "{}{}{}",
            escape_ident(&self.label),
            self.op,
            quote(&self.value)
        )
    }

    pub(crate) fn update_hash(&self, hasher: &mut Xxh3) {
        hasher.write(self.label.as_bytes());
        hasher.write(self.value.as_bytes());
        hasher.write(self.op.as_str().as_bytes())
    }
}

impl PartialEq<Self> for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        return self.op == other.op && self.label == other.label && self.value == other.value;
    }
}

impl PartialOrd for LabelFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Make sure the first label filter is __name__ (if any)
        if self.is_metric_name_filter() && !other.is_metric_name_filter() {
            return Some(Ordering::Less);
        }
        let mut order = self.label.cmp(&other.label);
        if order == Ordering::Equal {
            order = self.value.cmp(&other.value);
            if order == Ordering::Equal {
                order = self.op.to_string().cmp(&other.op.to_string());
            }
        }
        return Some(order);
    }
}

impl Ord for LabelFilter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl fmt::Display for LabelFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}{}",
            escape_ident(&self.label),
            self.op,
            quote(&self.value)
        )?;
        Ok(())
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatcherList(Vec<LabelFilter>);

impl MatcherList {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Deref for MatcherList {
    type Target = Vec<LabelFilter>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn remove_duplicate_label_filters(filters: &mut Vec<LabelFilter>) {
    let mut set: HashSet<String> = HashSet::with_capacity(filters.len());
    filters.retain(|filters| {
        let key = filters.to_string();
        if !set.contains(&key) {
            set.insert(key);
            true
        } else {
            false
        }
    })
}
