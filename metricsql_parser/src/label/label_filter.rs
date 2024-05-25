use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use ahash::AHashSet;
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

#[derive(
    Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash, Serialize, Deserialize,
)]
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
            _ => Err(ParseError::General(format!("Unexpected match op literal: {op}"))),
        }
    }
}

impl fmt::Display for LabelFilterOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Ord for LabelFilterOp {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(&other.as_str())
    }
}

/// LabelFilter represents MetricsQL label filter like `foo="bar"`.
#[derive(Default, Debug, Clone, Eq, Hash, Serialize, Deserialize)]
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

        assert!(!label.is_empty());
        let value = match match_op {
            LabelFilterOp::RegexEqual | LabelFilterOp::RegexNotEqual => {
                let label_value = value.into();
                let converted = try_escape_for_repeat_re(&label_value);
                let re_anchored = format!("^(?:{})$", converted);
                if compile_regexp(&re_anchored).is_err() {
                    return Err(ParseError::InvalidRegex(label_value));
                }
                converted
            }
            _ => value.into(),
        };

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
        // if we're matching against __name__, a negative comparison against the empty
        // string is valid
        let is_name_label = self.label == NAME_LABEL;
        match self.op {
            Equal => self.value.is_empty(),
            NotEqual => !self.value.is_empty() && !is_name_label,
            RegexEqual => is_empty_regex(&self.value),
            RegexNotEqual => is_empty_regex(&self.value) && !is_name_label,
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

    // Go and Rust handle the repeat pattern differently
    // in Go the following is valid: `aaa{bbb}ccc`
    // in Rust {bbb} is seen as an invalid repeat and must be ecaped \{bbb}
    // This escapes the opening "{" if it's not followed by valid repeat pattern (e.g. 4,6).
    fn try_parse_re(re: &str) -> Result<Regex, String> {
        Regex::new(re)
            .or_else(|_| Regex::new(&try_escape_for_repeat_re(re)))
            .map_err(|_| format!("illegal regex for {re}",))
    }
}

impl PartialEq<LabelFilter> for LabelFilter {
    fn eq(&self, other: &Self) -> bool {
        self.op.eq(&other.op) && self.label.eq(&other.label) && self.value.eq(&other.value)
    }
}

impl PartialOrd for LabelFilter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LabelFilter {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut cmp = self.label.cmp(&other.label);
        if cmp == Ordering::Equal {
            cmp = self.value.cmp(&other.value);
            if cmp == Ordering::Equal {
                cmp = self.op.cmp(&other.op);
            }
        }
        cmp
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

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matchers {
    pub matchers: Vec<LabelFilter>,
    pub or_matchers: Vec<Vec<LabelFilter>>,
}


impl Matchers {
    pub fn new(filters: Vec<LabelFilter>) -> Self {
        Matchers {
            matchers: filters,
            or_matchers: vec![],
        }
    }

    pub fn empty() -> Self {
        Self {
            matchers: vec![],
            or_matchers: vec![],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty() && self.or_matchers.is_empty()
    }

    pub fn with_or_matchers(mut self, or_matchers: Vec<Vec<LabelFilter>>) -> Self {
        self.or_matchers = or_matchers;
        self
    }

    pub fn append(mut self, matcher: LabelFilter) -> Self {
        // Check the latest or_matcher group. If it is not empty,
        // we need to add the current matcher to this group.
        let last_or_matcher = self.or_matchers.last_mut();
        if let Some(last_or_matcher) = last_or_matcher {
            last_or_matcher.push(matcher);
        } else {
            self.matchers.push(matcher);
        }
        self
    }

    pub fn append_or(mut self, matcher: LabelFilter) -> Self {
        if !self.matchers.is_empty() {
            // Be careful not to move ownership here, because it
            // will be used by the subsequent append method.
            let last_matchers = std::mem::take(&mut self.matchers);
            self.or_matchers.push(last_matchers);
        }
        let new_or_matchers = vec![matcher];
        self.or_matchers.push(new_or_matchers);
        self
    }

    /// Vector selectors must either specify a name or at least one label
    /// matcher that does not match the empty string.
    ///
    /// The following expression is illegal:
    /// {job=~".*"} # Bad!
    pub fn is_empty_matchers(&self) -> bool {
        (self.matchers.is_empty() && self.or_matchers.is_empty())
            || self
            .matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
            .all(|m| m.is_match(""))
    }

    /// find the matcher's value whose name equals the specified name. This function
    /// is designed to prepare error message of invalid promql expression.
    pub(crate) fn find_matcher_value(&self, name: &str) -> Option<String> {
        self.matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
            .find(|m| m.label.eq(name))
            .map(|m| m.value.clone())
    }

    /// find matchers whose name equals the specified name
    pub fn find_matchers(&self, name: &str) -> Vec<&LabelFilter> {
        self.matchers
            .iter()
            .chain(self.or_matchers.iter().flatten())
            .filter(|m| m.label.eq(name))
            .collect()
    }

    pub fn sort_filters(&mut self) {
        if !self.matchers.is_empty() {
            self.matchers.sort();
        }
        if !self.or_matchers.is_empty() {
            for filter_list in self.or_matchers.iter_mut() {
                filter_list.sort();
            }
        }
    }
}

impl fmt::Display for Matchers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let simple_matchers = &self.matchers;
        let or_matchers = &self.or_matchers;
        if or_matchers.is_empty() {
            write!(f, "{}", join_vector(simple_matchers, ",", true))
        } else {
            let or_matchers_string =
                self.or_matchers
                    .iter()
                    .fold(String::new(), |or_matchers_str, pair| {
                        format!("{} or {}", or_matchers_str, join_vector(pair, ", ", false))
                    });
            let or_matchers_string = or_matchers_string.trim_start_matches(" or").trim();
            write!(f, "{}", or_matchers_string)
        }
    }
}

pub fn remove_duplicate_label_filters(filters: &mut Vec<LabelFilter>) {
    let mut set: AHashSet<String> = AHashSet::with_capacity(filters.len());
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

// Go and Rust handle the repeat pattern differently
// in Go the following is valid: `aaa{bbb}ccc`
// in Rust {bbb} is seen as an invalid repeat and must be escaped \{bbb}
// This escapes the opening "{" if it's not followed by valid repeat pattern (e.g. 4,6).
pub fn try_escape_for_repeat_re(re: &str) -> String {
    fn is_repeat(chars: &mut std::str::Chars<'_>) -> (bool, String) {
        let mut buf = String::new();
        let mut comma_seen = false;
        for c in chars.by_ref() {
            buf.push(c);
            match c {
                ',' if comma_seen => {
                    return (false, buf); // ,, is invalid
                }
                ',' if buf == "," => {
                    return (false, buf); // {, is invalid
                }
                ',' if !comma_seen => comma_seen = true,
                '}' if buf == "}" => {
                    return (false, buf); // {} is invalid
                }
                '}' => {
                    return (true, buf);
                }
                _ if c.is_ascii_digit() => continue,
                _ => {
                    return (false, buf); // false if visit non-digit char
                }
            }
        }
        (false, buf) // not ended with }
    }

    let mut result = String::with_capacity(re.len() + 1);
    let mut chars = re.chars();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(cc) = chars.next() {
                    result.push(c);
                    result.push(cc);
                }
            }
            '{' => {
                let (is, s) = is_repeat(&mut chars);
                if !is {
                    result.push('\\');
                }
                result.push(c);
                result.push_str(&s);
            }
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};
    use regex::Regex;
    use crate::label::{LabelFilter, LabelFilterOp, Matchers};
    use super::try_escape_for_repeat_re;

    fn hash<H>(op: H) -> u64
        where
            H: Hash,
    {
        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_LabelFilterOp_hash() {
        assert_eq!(hash(LabelFilterOp::Equal), hash(LabelFilterOp::Equal));
        assert_eq!(hash(LabelFilterOp::NotEqual), hash(LabelFilterOp::NotEqual));
        assert_eq!(hash(LabelFilterOp::RegexEqual), hash(LabelFilterOp::RegexEqual));
        assert_eq!(hash(LabelFilterOp::RegexNotEqual), hash(LabelFilterOp::RegexNotEqual));
    }

    #[test]
    fn test_matcher_hash() {
        assert_eq!(
            hash(LabelFilter::new(LabelFilterOp::Equal, "name", "value").unwrap()),
            hash(LabelFilter::new(LabelFilterOp::Equal, "name", "value").unwrap()),
        );

        assert_eq!(
            hash(LabelFilter::new(LabelFilterOp::NotEqual, "name".into(), "value".into()).unwrap()),
            hash(LabelFilter::new(LabelFilterOp::NotEqual, "name", "value").unwrap()),
        );

        assert_eq!(
            hash(LabelFilter::regex_equal("name", "\\s+").unwrap()),
            hash(LabelFilter::regex_equal("name", "\\s+").unwrap()),
        );

        assert_eq!(
            hash(LabelFilter::regex_notequal("name", "\\s+").unwrap()),
            hash(LabelFilter::regex_notequal("name", "\\s+").unwrap()),
        );

        assert_ne!(
            hash(LabelFilter::new(LabelFilterOp::Equal, "name", "value").unwrap()),
            hash(LabelFilter::new(LabelFilterOp::NotEqual, "name", "value").unwrap()),
        );

        assert_ne!(
            hash(LabelFilter::regex_equal("name", "\\s+").unwrap()),
            hash(LabelFilter::regex_notequal("name", "\\s+").unwrap()),
        );
    }

    #[test]
    fn test_matcher_eq_ne() {
        let op = LabelFilterOp::Equal;
        let matcher = LabelFilter::new(op, "name", "up").unwrap();
        assert!(matcher.is_match("up"));
        assert!(!matcher.is_match("down"));

        let op = LabelFilterOp::NotEqual;
        let matcher = LabelFilter::new(op, "name", "up").unwrap();
        assert!(matcher.is_match("foo"));
        assert!(matcher.is_match("bar"));
        assert!(!matcher.is_match("up"));
    }

    #[test]
    fn test_matcher_re() {
        let value = "api/v1/.*";
        let matcher = LabelFilter::new(LabelFilterOp::RegexEqual, "name", value);
        assert!(matcher.is_match("api/v1/query"));
        assert!(matcher.is_match("api/v1/range_query"));
        assert!(!matcher.is_match("api/v2"));
    }

    #[test]
    fn test_eq_matcher_equality() {
        assert_eq!(
            LabelFilter::new(LabelFilterOp::Equal, "code", "200"),
            LabelFilter::new(LabelFilterOp::Equal, "code", "200")
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::Equal, "code", "200"),
            LabelFilter::new(LabelFilterOp::Equal, "code", "201")
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::Equal, "code", "200"),
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "200")
        );
    }

    #[test]
    fn test_ne_matcher_equality() {
        assert_eq!(
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "200"),
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "200")
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "200"),
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "201")
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::NotEqual, "code", "200"),
            LabelFilter::new(LabelFilterOp::Equal, "code", "200")
        );
    }

    #[test]
    fn test_re_matcher_equality() {
        assert_eq!(
            LabelFilter::regex_equal("code", "2??"),
            LabelFilter::regex_equal("code", "2??")
        );

        assert_ne!(
            LabelFilter::regex_equal("code", "2??",),
            LabelFilter::regex_equal("code", "2*?",)
        );

        assert_ne!(
            LabelFilter::new(LabelFilterOp::RegexEqual, "code", "2??",),
            LabelFilter::new(LabelFilterOp::Equal, "code", "2??")
        );
    }

    #[test]
    fn test_not_re_matcher_equality() {
        assert_eq!(
            LabelFilter::regex_notequal("code", "2??",),
            LabelFilter::regex_notequal("code", "2??",)
        );

        assert_ne!(
            LabelFilter::regex_notequal("code", "2??"),
            LabelFilter::regex_notequal("code", "2*?",)
        );

        assert_ne!(
            LabelFilter::regex_equal("code", "2??").unwrap(),
            LabelFilter::new(LabelFilterOp::Equal, "code", "2??").unwrap()
        );
    }

    #[test]
    fn test_matchers_equality() {
        assert_eq!(
            Matchers::empty()
                .append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap())
                .append(LabelFilter::new(LabelFilterOp::Equal, "name2", "val2").unwrap()),
            Matchers::empty()
                .append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap())
                .append(LabelFilter::new(LabelFilterOp::Equal, "name2", "val2").unwrap())
        );

        assert_ne!(
            Matchers::empty().append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap()),
            Matchers::empty().append(LabelFilter::new(LabelFilterOp::Equal, "name2", "val2").unwrap())
        );

        assert_ne!(
            Matchers::empty().append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap()),
            Matchers::empty().append(LabelFilter::new(LabelFilterOp::NotEqual, "name1", "val1").unwrap())
        );

        assert_eq!(
            Matchers::empty()
                .append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap())
                .append(LabelFilter::new(LabelFilterOp::NotEqual, "name2", "val2").unwrap())
                .append(LabelFilter::regex_equal("name2", "\\d+").unwrap())
                .append(LabelFilter::regex_notequal("name2", "\\d+").unwrap()),
            Matchers::empty()
                .append(LabelFilter::new(LabelFilterOp::Equal, "name1", "val1").unwrap())
                .append(LabelFilter::new(LabelFilterOp::NotEqual, "name2", "val2").unwrap())
                .append(LabelFilter::regex_equal("name2", "\\d+").unwrap())
                .append(LabelFilter::regex_notequal("name2", "\\d+").unwrap())
        );
    }

    #[test]
    fn test_find_matchers() {
        let matchers = Matchers::empty()
            .append(LabelFilter::new(LabelFilterOp::Equal, "foo".into(), "bar".into())
            .append(LabelFilter::new(LabelFilterOp::NotEqual, "foo", "bar"))
            .append(LabelFilter::new(LabelFilterOp::Equal, "FOO", "bar"))
            .append(LabelFilter::new(LabelFilterOp::NotEqual, "bar", "bar")));

        let ms = matchers.find_matchers("foo");
        assert_eq!(4, ms.len());
    }

    #[test]
    fn test_convert_re() {
        assert_eq!(try_escape_for_repeat_re("abc{}"), r"abc\{}");
        assert_eq!(try_escape_for_repeat_re("abc{def}"), r"abc\{def}");
        assert_eq!(try_escape_for_repeat_re("abc{def"), r"abc\{def");
        assert_eq!(try_escape_for_repeat_re("abc{1}"), "abc{1}");
        assert_eq!(try_escape_for_repeat_re("abc{1,}"), "abc{1,}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2}"), "abc{1,2}");
        assert_eq!(try_escape_for_repeat_re("abc{,2}"), r"abc\{,2}");
        assert_eq!(try_escape_for_repeat_re("abc{{1,2}}"), r"abc\{{1,2}}");
        assert_eq!(try_escape_for_repeat_re(r"abc\{abc"), r"abc\{abc");
        assert_eq!(try_escape_for_repeat_re("abc{1a}"), r"abc\{1a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,a}"), r"abc\{1,a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2a}"), r"abc\{1,2a}");
        assert_eq!(try_escape_for_repeat_re("abc{1,2,3}"), r"abc\{1,2,3}");
        assert_eq!(try_escape_for_repeat_re("abc{1,,2}"), r"abc\{1,,2}");
    }
}
