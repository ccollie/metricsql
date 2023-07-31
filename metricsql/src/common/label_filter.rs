use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use regex::Regex;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::Xxh3;

use crate::common::join_vector;
use crate::parser::{compile_regexp, escape_ident, is_empty_regex, quote, ParseError};

pub const NAME_LABEL: &str = "__name__";

pub type LabelName = String;

pub type LabelValue = String;

// NOTE: https://github.com/rust-lang/regex/issues/668
#[derive(Debug, Default, Clone)]
pub enum MatchOp {
    #[default]
    Equal,
    NotEqual,
    Re(Regex),
    NotRe(Regex),
}

impl MatchOp {
    pub fn is_negative(&self) -> bool {
        matches!(self, MatchOp::NotEqual | MatchOp::NotRe(_))
    }

    pub fn is_regex(&self) -> bool {
        matches!(self, Self::NotRe(_) | Self::Re(_))
    }
}

impl fmt::Display for MatchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchOp::Equal => write!(f, "="),
            MatchOp::NotEqual => write!(f, "!="),
            MatchOp::Re(reg) => write!(f, "=~{reg}"),
            MatchOp::NotRe(reg) => write!(f, "!~{reg}"),
        }
    }
}

impl PartialEq for MatchOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MatchOp::Equal, MatchOp::Equal) => true,
            (MatchOp::NotEqual, MatchOp::NotEqual) => true,
            (MatchOp::Re(s), MatchOp::Re(o)) => s.as_str().eq(o.as_str()),
            (MatchOp::NotRe(s), MatchOp::NotRe(o)) => s.as_str().eq(o.as_str()),
            _ => false,
        }
    }
}

impl Eq for MatchOp {}

impl Hash for MatchOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            MatchOp::Equal => "eq".hash(state),
            MatchOp::NotEqual => "ne".hash(state),
            MatchOp::Re(s) => format!("re:{}", s.as_str()).hash(state),
            MatchOp::NotRe(s) => format!("nre:{}", s.as_str()).hash(state),
        }
    }
}

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
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LabelFilter {
    pub op: LabelFilterOp,

    /// label contains label name for the filter.
    pub label: String,

    /// value contains unquoted value for the filter.
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
            RegexEqual => is_empty_regex(&self.value),
            RegexNotEqual => {
                let str = self.value.to_string();
                is_empty_regex(&str)
            }
        }
    }

    pub fn is_match(&self, str: &str) -> bool {
        match self.op {
            LabelFilterOp::Equal => self.value.eq(str),
            LabelFilterOp::NotEqual => self.value.ne(str),
            LabelFilterOp::RegexEqual => {
                // slight optimization for frequent case
                if str.is_empty() {
                    return is_empty_regex(&self.value);
                }
                if let Ok(re) = compile_regexp(&self.value) {
                    re.is_match(str)
                } else {
                    false
                }
            }
            LabelFilterOp::RegexNotEqual => {
                if let Ok(re) = compile_regexp(&self.value) {
                    !re.is_match(str)
                } else {
                    false
                }
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

    pub fn name(&self) -> String {
        if self.label == NAME_LABEL {
            return self.value.to_string();
        }
        self.label.clone()
    }

    pub(crate) fn update_hash(&self, hasher: &mut Xxh3) {
        hasher.write(self.label.as_bytes());
        hasher.write(self.value.as_bytes());
        hasher.write(self.op.as_str().as_bytes())
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
        Some(order)
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
pub struct Matchers(Vec<LabelFilter>);

impl Matchers {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// find the matcher's value whose name equals the specified name. This function
    /// is designed to prepare error message of invalid promql expression.
    pub fn find_matcher_value(&self, name: &str) -> Option<String> {
        for m in &self.0 {
            if m.label.eq(name) {
                return Some(m.value.clone());
            }
        }
        None
    }

    /// find matchers whose name equals the specified name
    pub fn find_matchers(&self, name: &str) -> Vec<LabelFilter> {
        self.0
            .iter()
            .filter(|m| m.label.eq(name))
            .cloned()
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, m: LabelFilter) {
        self.0.push(m);
    }

    pub fn sort(&mut self) {
        self.0.sort();
    }

    pub fn iter(&self) -> impl Iterator<Item = &LabelFilter> {
        self.0.iter()
    }
}

impl fmt::Display for Matchers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", join_vector(&self.0, ",", true))
    }
}

impl Deref for Matchers {
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
