use std::cmp::Ordering;
use std::fmt;
use enquote::enquote;
use regex::Regex;
use crate::error::{Error, Result};
use crate::lexer::{escape_ident, quote};
use crate::parser::compile_regexp_anchored;
use crate::types::StringExpr;

const NAME_LABEL: &str = "__name__";

pub type LabelName = String;

pub type LabelValue = String;

#[derive(Debug, Clone)]
pub enum LabelFilterOp {
    Equal,
    NotEqual,
    RegexEqual,
    RegexNotEqual
}

impl LabelFilterOp {
    pub fn is_negative(&self) -> bool {
        match self {
            LabelFilterOp::NotEqual | LabelFilterOp::RegexNotEqual => true,
            _ => false
        }
    }

    pub fn is_regex(&self) -> bool {
        match self {
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual => true,
            _ => false
        }
    }
}

impl TryFrom<&str> for LabelFilterOp {
    type Error = Error;

    fn try_from(op: &str) -> Result<Self> {
        match op {
            "=" => Ok(LabelFilterOp::Equal),
            "!=" => Ok(LabelFilterOp::NotEqual),
            "=~" => Ok(LabelFilterOp::RegexEqual),
            "!~" => Ok(LabelFilterOp::RegexNotEqual),
            _ => Err(Error::new("Unexpected match op literal")),
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

// LabelFilter represents MetricsQL label filter like `foo="bar"`.
#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct LabelFilter {
    pub op: LabelFilterOp,

    // Label contains label name for the filter.
    pub label: String,

    // Value contains unquoted value for the filter.
    pub value: String,

    re: Option<Regex>,
}

impl LabelFilter {
    pub fn new<N, V>(match_op: LabelFilterOp, label: N, value: V) -> Result<Self>
        where
            N: Into<LabelName>,
            V: Into<LabelValue>,
    {
        let label = label.into();
        let value = value.into();

        assert!(!label.is_empty());

        let re = match match_op {
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual => {
                compile_regexp_anchored(value.as_str()).map_err(|e| format!("{}", e)?)
            }
            _ => None,
        };

        Ok(Self {
            label,
            op: match_op,
            value,
            re,
        })
    }

    pub fn equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter> {
        LabelFilter::new(LabelFilterOp::Equal, key, value)
    }

    pub fn not_equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter> {
        LabelFilter::new(LabelFilterOp::NotEqual, key, value)
    }

    pub fn regex_equal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter> {
        LabelFilter::new(LabelFilterOp::RegexEqual, key, value)
    }

    pub fn regex_notequal<S: Into<String>>(key: S, value: S) -> Result<LabelFilter> {
        LabelFilter::new(LabelFilterOp::RegexNotEqual, key, value)
    }

    // IsRegexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op == LabelFilterOp::RegexEqual || self.op == LabelFilterOp::RegexNotEqual
    }

    // IsNegative represents whether the filter is negative, i.e. '!=' or '!~'.
    pub fn is_negative(&self) -> bool {
        self.op == LabelFilterOp::RegexEqual || self.op == LabelFilterOp::NotEqual
    }

    pub fn is_metric_name_filter(&self) -> bool {
        return self.label == NAME_LABEL && self.op == LabelFilterOp::Equal;
    }

    pub fn matches(&self, v: &str) -> bool {
        match self.match_op {
            LabelFilterOp::Equal => self.value == v,
            LabelFilterOp::NotEqual => self.value != v,
            LabelFilterOp::RegexEqual => self
                .re
                .as_ref()
                .expect("some regex is always expected for this type of matcher")
                .is_match(v),
            LabelFilterOp::RegexNotEqual => !self
                .re
                .as_ref()
                .expect("some regex is always expected for this type of matcher")
                .is_match(v),
        }
    }
}

impl PartialEq for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.op == other.op && self.value == other.value
    }
}

impl PartialOrd for LabelFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.key == other.keey {
            return Some(self.value.cmp(&other.value));
        }
        Some(self.key.cmp(&other.key))
    }
}


impl fmt::Display for LabelFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", escape_ident(&self.label) )?;
        write!(f, "{}", self.op)?;
        write!(f, "{}", quote( &self.value) )?;
        Ok(())
    }
}

// labelFilterExpr represents `foo <op> "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
//
// This type isn't exported.
#[derive(Default, Debug, Clone, PartialEq)]
pub(crate) struct LabelFilterExpr {
    pub label: String,
    pub value: StringExpr,
    pub op: LabelFilterOp
}

impl LabelFilterExpr {
    pub fn new<K: Into<String>>(label: K, value: StringExpr, op: LabelFilterOp) -> LabelFilterExpr {
        LabelFilterExpr {
            label: label.into(),
            value,
            op
        }
    }

    pub fn to_label_filter(&self) -> LabelFilter {
        if self.value.s.len() > 0 || self.value.tokens.len() > 0 {
            panic!("BUG: value must be already expanded; got {}", self.value)
        }
        // // Verify regexp.
        // if _, err := CompileRegexpAnchored(lfe.Value.S); err != nil {
        //   return Err(Error::new("invalid regexp in {}={}: {}", lf.label, lf.value, err)
        // }
        LabelFilter {
            label: self.label.to_string(),
            value: self.value.to_string(),
            op: self.op.clone(),
            re: None
        }
    }
}

impl fmt::Display for LabelFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}", escape_ident(&self.label), self.op, enquote( '\"', &self.value.s))?;
        Ok(())
    }
}
