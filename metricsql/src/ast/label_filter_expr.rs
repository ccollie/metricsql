use crate::lexer::{escape_ident};
use crate::parser::{compile_regexp, ParseError, ParseResult};
use std::fmt;
use crate::ast::{LabelFilter, LabelFilterOp, LabelName, LabelValue, NAME_LABEL};
use serde::{Serialize, Deserialize};
use crate::ast::segmented_string::SegmentedString;


/// LabelFilterExpr represents `foo <op> ident + "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
/// For internal use only, in the context of WITH expressions
#[derive(Default, Debug, Clone, Hash, Serialize, Deserialize)]
pub struct LabelFilterExpr {
    pub op: LabelFilterOp,

    /// Label contains label name for the filter.
    pub label: String,

    /// Value contains unquoted value for the filter.
    pub value: SegmentedString,
}

impl LabelFilterExpr {
    pub fn new<N, V>(match_op: LabelFilterOp, label: N, value: V) -> ParseResult<Self>
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
                return Err(ParseError::InvalidRegex(value))
            }
        }

        let value = SegmentedString::from(value.into());
        Ok(Self {
            label,
            op: match_op,
            value,
        })
    }

    pub fn equal<S: Into<String>>(key: S, value: S) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(LabelFilterOp::Equal, key, value)
    }

    pub fn not_equal<S: Into<String>>(key: S, value: S) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(LabelFilterOp::NotEqual, key, value)
    }

    /// is_regexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op.is_regex()
    }

    pub fn is_metric_name_filter(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    pub fn set_value<S: Into<String>>(&mut self, value: S) {
        self.value.set_from_string(value.into())
    }

    pub fn as_string(&self) -> String {
        format!("{}{}{}", escape_ident(&self.label), self.op, self.value)
    }

    pub fn is_resolved(&self) -> bool {
        self.value.is_expanded()
    }

    pub fn to_label_filter(&self) -> ParseResult<LabelFilter> {
        LabelFilter::new(self.op, &self.label, self.value.value.to_string())
    }
}

impl PartialEq for LabelFilterExpr {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.op == other.op && self.value == other.value
    }
}

impl fmt::Display for LabelFilterExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}", escape_ident(&self.label), self.op, &self.value)?;
        Ok(())
    }
}