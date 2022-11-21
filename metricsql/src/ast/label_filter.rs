use crate::lexer::{escape_ident, quote, TextSpan};
use crate::parser::{compile_regexp, ParseError};
use enquote::enquote;
use std::fmt;
use crate::ast::StringExpr;
use serde::{Serialize, Deserialize};

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
        matches!(self, LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual)
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
            if compile_regexp(&re_anchored).is_err() { return Err(ParseError::InvalidRegex(value)) }
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

// impl PartialOrd for LabelFilter {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         if self.label == other.label {
//             return Some(self.value.cmp(&other.value));
//         }
//         Some(self.value.cmp(&other.value))
//     }
// }

impl fmt::Display for LabelFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_string())?;
        Ok(())
    }
}


/// labelFilterExpr represents `foo <op> "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
///
/// This type isn't exported.
#[derive(Default, Debug, Clone, Hash, Serialize, Deserialize)]
pub(crate) struct LabelFilterExpr {
    pub label: String,
    pub value: StringExpr,
    pub op: LabelFilterOp,
    init: bool
}

impl LabelFilterExpr {
    pub fn new<K: Into<String>>(label: K, value: StringExpr, op: LabelFilterOp) -> Self {
        LabelFilterExpr {
            label: label.into(),
            value,
            op,
            init: true
        }
    }

    pub fn new_tag<S: Into<String>, TS: Into<TextSpan>>(label: S, op: LabelFilterOp, value: S, span: TS) -> Self {
        LabelFilterExpr {
            label: label.into(),
            value: StringExpr::new(value.into(), span.into()),
            op,
            init: true
        }
    }

    pub(crate) fn is_init(&self) -> bool {
        self.init
    }

    pub fn to_label_filter(&self) -> LabelFilter {
        if !self.is_expanded() {
            panic!("BUG: value must be already expanded; got {}", self.value)
        }
        // // Verify regexp.
        // if _, err := CompileRegexpAnchored(lfe.Value.S); err != nil {
        //   return Err(Error::new("invalid regexp in {}={}: {}", lf.label, lf.value, err)
        // }
        LabelFilter {
            label: self.label.to_string(),
            value: self.value.value.to_string(),
            op: self.op,
        }
    }

    pub fn is_expanded(&self) -> bool {
        !self.value.is_empty() && !self.value.has_tokens()
    }
}

impl fmt::Display for LabelFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}{}",
            escape_ident(&self.label),
            self.op,
            enquote('\"', &self.value.value)
        )?;
        Ok(())
    }
}
