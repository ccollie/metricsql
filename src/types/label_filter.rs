use std::fmt;
use regex::Regex;

use crate::error::{Error, Result};
use crate::StringExpr;
use enquote::enquote;

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

impl std::convert::TryFrom<&str> for LabelFilterOp {
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
#[derive(Default, Debug, Clone, PartialEq)]
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
                Some(Regex::new(&format!("^(?:{})$", value)).map_err(|e| format!("{}", e))?)
            }
            _ => None,
        };

        Ok(Self {
            label,
            op,
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
        return self.label == "__name__" && self.op == LabelFilterOp::Equal;
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


impl fmt::Display for LabelFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", escaped_ident(&self.label) )?;
        write!(f, "{}", self.op)?;
        write!( enquote( '\"', &self.value) );
        Ok(())
    }
}


// labelFilterExpr represents `foo <op> "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
//
// This type isn't exported.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct LabelFilterExpr {
    pub label: string,
    pub value: StringExpr,
    pub is_regexp: bool,
    pub is_negative: bool
}

impl LabelFilterExpr {
    pub fn new(label: string, value: &StringExpr, isRegexp: bool, isNegative: bool) -> LabelFilterExpr {
        LabelFilterExpr {
            label,
            value,
            is_regexp: isRegexp,
            is_negative: isNegative
        }
    }

    pub fn to_label_filter(&self) -> LabelFilter {
        if self.value == nil || lfe.value.tokens.len() > 0 {
            panic(fmt.Errorf("BUG: lfe.Value must be already expanded; got %v", lfe.Value))
        }
        // // Verify regexp.
        // if _, err := CompileRegexpAnchored(lfe.Value.S); err != nil {
        // return nil, fmt.Errorf("invalid regexp in %s=%q: %s", lf.Label, lf.Value, err)
        // }
        LabelFilter {
            label: self.label.to_string(),
            value: self.value.to_string(),
            is_regexp: self.is_regexp,
            is_negative: self.is_negative
        }
    }
}

impl fmt::Display for LabelFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut dst: &String = String::new( escape_ident(lf.Label) );
        let op: String;

        if lf.isNegative {
            if self.is_regexp {
                op = "!~".parse().unwrap()
            } else {
                op = "!="
            }
        } else {
            if self.is_regexp {
                op = "=~"
            } else {
                op = "="
            }
        }
        dst.push_str(op);
        dst.push_str( enquote( '\"', self.value) );
        Ok(())
    }
}
