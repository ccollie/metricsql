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

/// Important! this is constructed from a regex
/// so the literals MUST be ordered by their order
/// in the regex pattern
#[derive(Clone, Debug)]
pub struct ContainsAnyOfHandler {
    pub literals: Vec<String>,
    pub suffix: String,
}

impl PredicateReflection for ContainsAnyOfHandler {}

impl Display for ContainsAnyOfHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "value contains one of [{:?}]", self.literals)
    }
}

impl Predicate<str> for ContainsAnyOfHandler {
    fn eval(&self, variable: &str) -> bool {
        if !self.suffix.is_empty() && !variable.ends_with(&self.suffix) {
            return false;
        }
        let mut n = 0;
        let mut cursor = &variable[0..];
        while let Some(pos) = cursor.find(&self.literals[n]) {
            n += 1;
            if n == self.literals.len() {
                return true;
            }
            cursor = &cursor[pos + 1..];
        }
        true
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
pub struct OrStringMatcher {
    pub left: Box<StringMatchHandler>,
    pub right: Box<StringMatchHandler>,
}

impl OrStringMatcher {
    pub fn new(left: StringMatchHandler, right: StringMatchHandler) -> Self {
        Self {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

impl Display for OrStringMatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} or {})", self.left, self.right)
    }
}

impl PredicateReflection for OrStringMatcher {}
impl Predicate<str> for OrStringMatcher {
    fn eval(&self, variable: &str) -> bool {
        self.left.eval(variable) || self.right.eval(variable)
    }
}

#[derive(Debug, Clone)]
pub enum StringMatchHandler {
    AlwaysTrue,
    Fsm(FastStringMatcher),
    FastRegex(FastRegexMatcher),
    Alternates(Vec<String>),
    ContainsAnyOf(ContainsAnyOfHandler),
    MatchFn(MatchFnHandler),
    Or(OrStringMatcher),
    Regex(Regex),
}

impl Default for StringMatchHandler {
    fn default() -> Self {
        Self::dot_plus()
    }
}

impl StringMatchHandler {
    pub fn literal<T: Into<String>>(value: T) -> Self {
        Self::MatchFn(MatchFnHandler::new(value, matches_literal))
    }

    #[allow(dead_code)]
    pub fn literal_mismatch<T: Into<String>>(value: T) -> Self {
        Self::MatchFn(MatchFnHandler::new(value, mismatches_literal))
    }

    #[allow(dead_code)]
    pub fn alternates(alts: Vec<String>) -> Self {
        Self::Alternates(alts)
    }

    /// handler for .*
    #[allow(dead_code)]
    pub fn dot_star() -> Self {
        Self::MatchFn(MatchFnHandler::new("", dot_star))
    }

    /// handler for .+
    pub fn dot_plus() -> Self {
        Self::MatchFn(MatchFnHandler::new("", dot_plus))
    }

    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    pub fn starts_with<T: Into<String>>(prefix: T) -> Self {
        Self::MatchFn(MatchFnHandler::new(prefix.into(), starts_with))
    }

    #[allow(dead_code)]
    pub fn contains<T: Into<String>>(needle: T) -> Self {
        Self::MatchFn(MatchFnHandler::new(needle.into(), contains))
    }

    #[allow(dead_code)]
    pub fn prefix<T: Into<String>>(prefix: T, is_dot_star: bool) -> Self {
        Self::MatchFn(MatchFnHandler::new(
            prefix,
            if is_dot_star {
                prefix_dot_star
            } else {
                prefix_dot_plus
            },
        ))
    }

    #[allow(dead_code)]
    pub fn not_prefix<T: Into<String>>(prefix: T, is_dot_star: bool) -> Self {
        Self::MatchFn(MatchFnHandler::new(
            prefix,
            if is_dot_star {
                not_prefix_dot_star
            } else {
                not_prefix_dot_plus
            },
        ))
    }

    #[allow(dead_code)]
    pub fn suffix<T: Into<String>>(suffix: T, is_dot_star: bool) -> Self {
        Self::MatchFn(MatchFnHandler::new(
            suffix,
            if is_dot_star {
                suffix_dot_star
            } else {
                suffix_dot_plus
            },
        ))
    }

    #[allow(dead_code)]
    pub fn not_suffix<T: Into<String>>(suffix: T, is_dot_star: bool) -> Self {
        Self::MatchFn(MatchFnHandler::new(
            suffix,
            if is_dot_star {
                not_suffix_dot_star
            } else {
                not_suffix_dot_plus
            },
        ))
    }

    #[allow(dead_code)]
    pub(super) fn middle(prefix: &'static str, pattern: String, suffix: &'static str) -> Self {
        match (prefix, suffix) {
            (".+", ".+") => Self::match_fn(pattern, dot_plus_dot_plus),
            (".*", ".*") => Self::match_fn(pattern, dot_star_dot_star),
            (".*", ".+") => Self::match_fn(pattern, dot_star_dot_plus),
            (".+", ".*") => Self::match_fn(pattern, dot_plus_dot_star),
            _ => unreachable!("Invalid prefix and suffix combination"),
        }
    }

    #[allow(dead_code)]
    pub(super) fn not_middle(prefix: &'static str, pattern: String, suffix: &'static str) -> Self {
        match (prefix, suffix) {
            (".+", ".+") => Self::match_fn(pattern, not_dot_plus_dot_plus),
            (".+", ".*") => Self::match_fn(pattern, not_dot_plus_dot_star),
            (".*", ".+") => Self::match_fn(pattern, not_dot_star_dot_plus),
            (".*", ".*") => Self::match_fn(pattern, not_dot_star_dot_star),
            _ => unreachable!("Invalid prefix and suffix combination"),
        }
    }

    #[allow(dead_code)]
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::AlwaysTrue => true,
            StringMatchHandler::Alternates(alts) => matches_alternates(alts, s),
            StringMatchHandler::Fsm(fsm) => fsm.matches(s),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::FastRegex(r) => r.matches(s),
            StringMatchHandler::ContainsAnyOf(m) => m.eval(s),
            StringMatchHandler::Or(p) => p.eval(s),
            StringMatchHandler::Regex(r) => r.is_match(s),
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

fn starts_with(prefix: &str, candidate: &str) -> bool {
    candidate.starts_with(prefix)
}

#[allow(dead_code)]
fn contains(prefix: &str, candidate: &str) -> bool {
    candidate.contains(prefix)
}

#[allow(dead_code)]
fn matches_alternates(or_values: &[String], s: &str) -> bool {
    or_values.iter().any(|v| v == s)
}

#[allow(dead_code)]
fn matches_literal(prefix: &str, candidate: &str) -> bool {
    prefix == candidate
}

#[allow(dead_code)]
fn mismatches_literal(prefix: &str, candidate: &str) -> bool {
    prefix != candidate
}

// .*
fn dot_star(_: &str, _: &str) -> bool {
    true
}

// .+
fn dot_plus(_: &str, candidate: &str) -> bool {
    !candidate.is_empty()
}

// prefix + '.*'
#[allow(dead_code)]
fn prefix_dot_star(prefix: &str, candidate: &str) -> bool {
    // Fast path - the pr contains "prefix.*"
    candidate.starts_with(prefix)
}

#[allow(dead_code)]
fn not_prefix_dot_star(prefix: &str, candidate: &str) -> bool {
    !candidate.starts_with(prefix)
}

// prefix.+'
#[allow(dead_code)]
fn prefix_dot_plus(prefix: &str, candidate: &str) -> bool {
    // dot plus
    candidate.len() > prefix.len() && candidate.starts_with(prefix)
}

#[allow(dead_code)]
fn not_prefix_dot_plus(prefix: &str, candidate: &str) -> bool {
    candidate.len() <= prefix.len() || !candidate.starts_with(prefix)
}

// suffix.*'
#[allow(dead_code)]
fn suffix_dot_star(suffix: &str, candidate: &str) -> bool {
    // Fast path - the pr contains "prefix.*"
    candidate.ends_with(suffix)
}

#[allow(dead_code)]
fn not_suffix_dot_star(suffix: &str, candidate: &str) -> bool {
    !candidate.ends_with(suffix)
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
fn not_suffix_dot_plus(suffix: &str, candidate: &str) -> bool {
    if candidate.len() <= suffix.len() {
        true
    } else {
        let temp = skip_last_char(candidate);
        temp != suffix
    }
}

#[allow(dead_code)]
fn dot_star_dot_star(pattern: &str, candidate: &str) -> bool {
    candidate.contains(pattern)
}

#[allow(dead_code)]
fn not_dot_star_dot_star(pattern: &str, candidate: &str) -> bool {
    !candidate.contains(pattern)
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

#[allow(dead_code)]
fn not_dot_plus_dot_star(pattern: &str, candidate: &str) -> bool {
    if candidate.len() <= pattern.len() {
        true
    } else {
        let temp = skip_first_char(candidate);
        !temp.contains(pattern)
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

#[allow(dead_code)]
fn not_dot_star_dot_plus(pattern: &str, candidate: &str) -> bool {
    if candidate.len() <= pattern.len() {
        true
    } else {
        let temp = skip_last_char(candidate);
        !temp.contains(pattern)
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

#[allow(dead_code)]
fn not_dot_plus_dot_plus(pattern: &str, candidate: &str) -> bool {
    if candidate.len() <= pattern.len() + 1 {
        true
    } else {
        let sub = skip_first_and_last_char(candidate);
        !sub.contains(pattern)
    }
}
