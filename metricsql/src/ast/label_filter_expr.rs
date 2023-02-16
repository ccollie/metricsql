use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ast::StringExpr;
use crate::common::{LabelFilter, LabelFilterOp, LabelName, NAME_LABEL};
use crate::lexer::escape_ident;
use crate::parser::{compile_regexp, ParseError, ParseResult};

/// LabelFilterExpr represents `foo <op> ident + "bar"` expression, where <op> is `=`, `!=`, `=~` or `!~`.
/// For internal use only, in the context of WITH expressions
#[derive(Default, Debug, Clone, Hash, Serialize, Deserialize)]
pub struct LabelFilterExpr {
    pub op: LabelFilterOp,

    /// Label contains label name for the filter.
    pub label: String,

    /// Value contains unquoted value for the filter.
    pub value: StringExpr,
}

impl LabelFilterExpr {
    pub fn new<N>(match_op: LabelFilterOp, label: N, value: StringExpr) -> ParseResult<Self>
    where
        N: Into<LabelName>,
    {
        let label = label.into();

        assert!(!label.is_empty());

        if match_op == LabelFilterOp::RegexEqual || match_op == LabelFilterOp::RegexNotEqual {
            if value.is_resolved() {
                let resolved_value = value.to_string();
                let re_anchored = format!("^(?:{})$", resolved_value);
                if compile_regexp(&re_anchored).is_err() {
                    return Err(ParseError::InvalidRegex(resolved_value));
                }
            }
        }

        let value = StringExpr::from(value.to_string());
        Ok(Self {
            label,
            op: match_op,
            value,
        })
    }

    pub fn equal<S: Into<String>>(key: S, value: StringExpr) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(LabelFilterOp::Equal, key, value)
    }

    pub fn not_equal<S: Into<String>>(key: S, value: StringExpr) -> ParseResult<LabelFilterExpr> {
        LabelFilterExpr::new(LabelFilterOp::NotEqual, key, value)
    }

    /// is_negative represents whether the filter is negative, i.e. '!=' or '!~'.
    pub fn is_negative(&self) -> bool {
        self.op.is_negative()
    }

    /// is_regexp represents whether the filter is regexp, i.e. `=~` or `!~`.
    pub fn is_regexp(&self) -> bool {
        self.op.is_regex()
    }

    pub fn is_metric_name_filter(&self) -> bool {
        self.label == NAME_LABEL && self.op == LabelFilterOp::Equal
    }

    pub fn set_value<S: Into<String>>(&mut self, value: S) {
        self.value = StringExpr::from(value.into())
    }

    pub fn as_string(&self) -> String {
        format!("{}{}{}", escape_ident(&self.label), self.op, self.value)
    }

    pub fn is_resolved(&self) -> bool {
        self.value.is_resolved()
    }

    pub fn to_label_filter(&self) -> ParseResult<LabelFilter> {
        LabelFilter::new(self.op, &self.label, self.value.to_string())
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
