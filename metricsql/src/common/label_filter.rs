use crate::parser::{compile_regexp, escape_ident, is_empty_regex, ParseError, quote};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Deref;

pub const NAME_LABEL: &str = "__name__";

pub type LabelName = String;

pub type LabelValue = String;

#[derive(Default, Debug, Clone, PartialEq, Eq, Copy, Hash, Serialize, Deserialize)]
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
        match self {
            LabelFilterOp::Equal => write!(f, "="),
            LabelFilterOp::NotEqual => write!(f, "!="),
            LabelFilterOp::RegexEqual => write!(f, "=~"),
            LabelFilterOp::RegexNotEqual => write!(f, "!~"),
        }
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
            },
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
}

impl PartialEq for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.op == other.op && self.value == other.value
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
