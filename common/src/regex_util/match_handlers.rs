use crate::bytes_util::{FastRegexMatcher, FastStringMatcher};
use predicates::reflection::PredicateReflection;
use predicates::Predicate;
use regex::Regex;
use std::fmt::{Display, Formatter};

use super::regex_utils::{skip_first_and_last_char, skip_first_char, skip_last_char};

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;

#[derive(Clone, Debug)]
pub struct AlternateMatchHandler(pub Vec<String>);

impl PredicateReflection for AlternateMatchHandler {}

impl Display for AlternateMatchHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "value in [{:?}]", self.0)
    }
}

impl Predicate<str> for AlternateMatchHandler {
    fn eval(&self, variable: &str) -> bool {
        matches_alternates(&self.0, variable)
    }
}

/// Important! this is constructed from a regex so the literals MUST be ordered by their order
/// in the regex pattern
#[derive(Clone, Debug)]
pub struct ContainsAnyOfPredicate {
    pub literals: Vec<String>,
}

impl ContainsAnyOfPredicate {
    pub fn new(literals: Vec<String>) -> Self {
        Self { literals }
    }

    pub fn is_match(&self, variable: &str) -> bool {
        if self.literals.is_empty() {
            return false;
        }
        let mut cursor = &variable[0..];
        for literal in self.literals.iter() {
            if let Some(pos) = cursor.find(literal) {
                cursor = &cursor[pos + 1..];
            } else {
                return false;
            }
        }
        true
    }
}

impl PredicateReflection for ContainsAnyOfPredicate {}

impl Display for ContainsAnyOfPredicate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "value contains one of [{:?}]", self.literals)
    }
}

impl Predicate<str> for ContainsAnyOfPredicate {
    fn eval(&self, variable: &str) -> bool {
        self.is_match(variable)
    }
}

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

#[derive(Clone, Debug)]
pub enum StringMatchHandler {
    MatchAll,
    MatchNone,
    Fsm(FastStringMatcher),
    FastRegex(FastRegexMatcher),
    Alternates(Vec<String>),
    ContainsAnyOf(ContainsAnyOfPredicate),
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
    pub fn alternates(alts: Vec<String>) -> Self {
        Self::Alternates(alts)
    }

    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    #[allow(dead_code)]
    pub fn contains<T: Into<String>>(needle: T) -> Self {
        Self::MatchFn(MatchFnHandler::new(needle.into(), |needle, haystack| {
            haystack.contains(needle)
        }))
    }

    pub fn contains_any_of(literals: Vec<String>) -> Self {
        Self::ContainsAnyOf(ContainsAnyOfPredicate::new(literals))
    }

    pub fn and(self, b: StringMatchHandler) -> Self {
        Self::And(Box::new(self), Box::new(b))
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::MatchAll => true,
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::Alternates(alts) => matches_alternates(alts, s),
            StringMatchHandler::Fsm(fsm) => fsm.matches(s),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::FastRegex(r) => r.matches(s),
            StringMatchHandler::ContainsAnyOf(m) => m.eval(s),
            StringMatchHandler::Regex(r) => r.is_match(s),
            StringMatchHandler::And(a, b) => a.matches(s) && b.matches(s),
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

#[allow(dead_code)]
fn matches_alternates(or_values: &[String], s: &str) -> bool {
    or_values.iter().any(|v| s.contains(v))
}

// prefix + '.*'
#[allow(dead_code)]
pub(super) fn match_prefix_dot_star(prefix: &str, candidate: &str) -> bool {
    // Fast path - the pr contains "prefix.*"
    candidate.starts_with(prefix)
}

// prefix.+'
#[allow(dead_code)]
pub(super) fn match_prefix_dot_plus(prefix: &str, candidate: &str) -> bool {
    // dot plus
    candidate.len() > prefix.len() && candidate.starts_with(prefix)
}

// suffix.*'
#[allow(dead_code)]
fn suffix_dot_star(suffix: &str, candidate: &str) -> bool {
    // Fast path - the pr contains "prefix.*"
    candidate.ends_with(suffix)
}

// suffix.+'
#[allow(dead_code)]
fn suffix_dot_plus(suffix: &str, candidate: &str) -> bool {
    // dot plus
    if candidate.len() > suffix.len() {
        let temp = skip_last_char(candidate);
        temp == suffix
    } else {
        false
    }
}

#[allow(dead_code)]
fn dot_star_dot_star(pattern: &str, candidate: &str) -> bool {
    candidate.contains(pattern)
}

// '.+middle.*'
#[allow(dead_code)]
fn dot_plus_dot_star(pattern: &str, candidate: &str) -> bool {
    if candidate.len() > pattern.len() {
        let temp = skip_first_char(candidate);
        temp.contains(pattern)
    } else {
        false
    }
}

// '.*middle.+'
#[allow(dead_code)]
fn dot_star_dot_plus(pattern: &str, candidate: &str) -> bool {
    if candidate.len() > pattern.len() {
        let temp = skip_last_char(candidate);
        temp.contains(pattern)
    } else {
        false
    }
}

// '.+middle.+'
#[allow(dead_code)]
fn dot_plus_dot_plus(pattern: &str, candidate: &str) -> bool {
    if candidate.len() > pattern.len() + 1 {
        let sub = skip_first_and_last_char(candidate);
        sub.contains(pattern)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alternates() {
        let handler = StringMatchHandler::alternates(vec!["a".to_string(), "b".to_string()]);
        assert!(handler.matches("a"));
        assert!(handler.matches("b"));
        assert!(!handler.matches("c"));
    }

    #[test]
    fn test_contains() {
        let handler = StringMatchHandler::contains("a");
        assert!(handler.matches("a"));
        assert!(handler.matches("ba"));
        assert!(!handler.matches("b"));
    }
}