use crate::bytes_util::{FastRegexMatcher, FastStringMatcher};
use predicates::reflection::PredicateReflection;
use predicates::Predicate;
use regex::Regex;
use std::fmt::{Display, Formatter};
use crate::prelude::regex_utils::skip_first_char;

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncludesAnyOfMatcher(pub Vec<String>);

impl PredicateReflection for IncludesAnyOfMatcher {}

impl Display for IncludesAnyOfMatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "value includes one of [{:?}]", self.0)
    }
}

impl Predicate<str> for IncludesAnyOfMatcher {
    fn eval(&self, variable: &str) -> bool {
        self.0.iter().any(|v| variable.contains(v))
    }
}

pub struct AlterMatchOptions {
    pub anchor_end: bool,
    pub anchor_start: bool,
    pub prefix_dot_plus: bool,
    pub suffix_dot_plus: bool,
}

#[derive(Clone, Debug)]
pub enum StringMatchHandler {
    MatchAll,
    MatchNone,
    Empty,
    NotEmpty,
    Literal(String),
    Contains(String),
    StartsWith(String),
    EndsWith(String),
    Fsm(FastStringMatcher),
    FastRegex(FastRegexMatcher),
    Alternates(Vec<String>, bool),
    OrderedAlternates(Vec<String>),
    MatchFn(MatchFnHandler),
    And(Box<StringMatchHandler>, Box<StringMatchHandler>),
    Regex(Regex),
}

impl Default for StringMatchHandler {
    fn default() -> Self {
        Self::MatchAll
    }
}

impl StringMatchHandler {
    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    pub fn fast_regex(regex: Regex) -> Self {
        Self::FastRegex(FastRegexMatcher::new(regex))
    }

    pub fn and(self, b: StringMatchHandler) -> Self {
        Self::And(Box::new(self), Box::new(b))
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::MatchAll => true,
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::Alternates(alts, match_end) => matches_alternates(alts, s, *match_end),
            StringMatchHandler::Fsm(fsm) => fsm.matches(s),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::FastRegex(r) => r.matches(s),
            StringMatchHandler::OrderedAlternates(m) => match_ordered_alternates(&m, s),
            StringMatchHandler::Regex(r) => r.is_match(s),
            StringMatchHandler::And(a, b) => a.matches(s) && b.matches(s),
            StringMatchHandler::Contains(value) => s.contains(value),
            StringMatchHandler::StartsWith(prefix) => s.starts_with(prefix),
            StringMatchHandler::EndsWith(suffix) => s.ends_with(suffix),
            StringMatchHandler::Literal(val) => s == val,
            StringMatchHandler::Empty => s.is_empty(),
            StringMatchHandler::NotEmpty => !s.is_empty(),
        }
    }
}

impl Predicate<str> for StringMatchHandler {
    fn eval(&self, variable: &str) -> bool {
        self.matches(variable)
    }
}

#[derive(Clone, Debug)]
pub struct MatchFnHandler {
    pattern: String,
    pub(super) match_fn: MatchFn,
}

impl MatchFnHandler {
    pub(super) fn new<T: Into<String>>(pattern: T, match_fn: MatchFn) -> Self {
        Self {
            pattern: pattern.into(),
            match_fn,
        }
    }

    #[allow(dead_code)]
    pub(super) fn matches(&self, s: &str) -> bool {
        (self.match_fn)(&self.pattern, s)
    }
}

impl PredicateReflection for StringMatchHandler {}

impl Display for StringMatchHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Predicate<&str> for StringMatchHandler {
    fn eval(&self, variable: &&str) -> bool {
        self.matches(variable)
    }
}

fn matches_alternates(or_values: &[String], s: &str, match_end: bool) -> bool {
    if match_end {
        for needle in or_values.iter() {
            if let Some(pos) = s.find(needle) {
                let found = pos + needle.len() < s.len() - 1;
                if found {
                    return true;
                }
            }
        }
        false
    } else {
        or_values.iter().any(|v| s.contains(v))
    }
}

fn match_alternates_exact(or_values: &[String], haystack: &str, options: AlterMatchOptions) -> bool {
    let mut haystack = &haystack[0..];
    if options.anchor_start && options.anchor_end {
        return match (options.prefix_dot_plus, options.suffix_dot_plus) {
            (true, true) => {
                // ^.+(foo|bar).+$
                or_values.iter().any(|v| match_prefix_dot_plus(v, haystack))
            }
            (true, false) => {
                // ^.+(foo|bar)$
                or_values.iter().any(|v| match_prefix_dot_plus(v, haystack))
            }
            (false, true) => {
                // ^(foo|bar).+$
                or_values.iter().any(|v| haystack.starts_with(v))
            }
            _ => {
                // ^(foo|bar)$
                or_values.iter().any(|v| haystack == v)
            }
        }
    } else if options.anchor_end {
        // (foo|bar)$
        return match (options.prefix_dot_plus, options.suffix_dot_plus) {
            (true, true) => {
                // .+(foo|bar)$
                if !haystack.is_empty() {
                    haystack = skip_first_char(haystack);
                }
                or_values.iter().any(|v| haystack.ends_with(v))
            }
            (true, false) => {
                // .+(foo|bar)$
                if !haystack.is_empty() {
                    haystack = skip_first_char(haystack);
                }
                or_values.iter().any(|v| match_prefix_dot_plus(v, haystack))
            }
            (false, true) => {
                // (foo|bar).+$
                or_values.iter().any(|v| haystack.contains(v))
            }
            _ => {
                // (foo|bar)$
                or_values.iter().any(|v| haystack.ends_with(v))
            }
        }
    } else  if options.anchor_start {
        return or_values.iter().any(|v| haystack.starts_with(v));
    }
    or_values.iter().any(|v| haystack == v)
}

fn match_ordered_alternates(or_values: &[String], s: &str) -> bool {
    if or_values.is_empty() {
        return false;
    }
    let mut cursor = &s[0..];
    for literal in or_values.iter() {
        if let Some(pos) = cursor.find(literal) {
            cursor = &cursor[pos + 1..];
        } else {
            return false;
        }
    }
    true
}

fn match_literal(value: &str, candidate: &str, anchor_start: bool, anchor_end: bool) -> bool {
    if anchor_start && anchor_end {
        candidate == value
    } else if anchor_start {
        candidate.starts_with(value)
    } else if anchor_end {
        candidate.ends_with(value)
    } else {
        candidate.contains(value)
    }
}

// prefix.+'
pub(super) fn match_prefix_dot_plus(prefix: &str, candidate: &str) -> bool {
    // dot plus
    candidate.len() > prefix.len() && candidate.starts_with(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alternates() {
        let handler = StringMatchHandler::Alternates(vec!["a".to_string(), "b".to_string()], false);
        assert!(handler.matches("a"));
        assert!(handler.matches("b"));
        assert!(!handler.matches("c"));
    }

    #[test]
    fn test_contains() {
        let handler = StringMatchHandler::Contains("a".to_string());
        assert!(handler.matches("a"));
        assert!(handler.matches("ba"));
        assert!(!handler.matches("b"));
    }
}