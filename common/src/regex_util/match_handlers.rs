use crate::bytes_util::{FastRegexMatcher, FastStringMatcher};
use predicates::reflection::PredicateReflection;
use predicates::Predicate;
use regex::Regex;
use std::fmt::{Display, Formatter};

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;
pub type AlternatesMatchFn = fn(or_values: &[String], haystack: &str) -> bool;

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

pub struct StringMatchOptions {
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
    AlternatesFn(Vec<String>, AlternatesMatchFn),
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

    pub fn alternates(alts: Vec<String>, options: &StringMatchOptions) -> Self {
        if alts.len() == 1 {
            let mut alts = alts;
            let pattern = alts.pop().unwrap();
            let match_fn = get_literal_match_fn(options);
            return Self::MatchFn(MatchFnHandler{
                pattern,
                match_fn
            });
        }
        let match_fn = get_alternate_match_fn(options);
        Self::AlternatesFn(alts, match_fn)
    }

    pub fn literal(value: String, options: &StringMatchOptions) -> Self {
        let match_fn = get_literal_match_fn(options);
        Self::MatchFn(MatchFnHandler::new(value, match_fn))
    }

    pub fn equals(value: String) -> Self {
        Self::MatchFn(MatchFnHandler::new(value, equals_fn))
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
            StringMatchHandler::AlternatesFn(alternates, handler) => handler(alternates, s),
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

fn get_alternate_match_fn(options: &StringMatchOptions) -> AlternatesMatchFn {

    fn contains(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| haystack.contains(v))
    }

    fn starts_with(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| haystack.starts_with(v))
    }

    fn starts_with_dot_plus(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| haystack.len() > v.len() && haystack.starts_with(v))
    }

    fn ends_with(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| haystack.ends_with(v))
    }

    fn equals(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| haystack == v)
    }

    fn dot_plus_dot_plus(alternates: &[String], haystack: &str) -> bool {
        if haystack.len() < 2 {
            return false;
        }
        alternates.iter().any(|needle| dot_plus_dot_plus_fn(needle, haystack))
    }

    // ^.+(foo|bar)$ / .+(foo|bar)$
    fn dot_plus_ends_with(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| dot_plus_ends_with_fn(v, haystack))
    }

    fn prefix_dot_plus_(alternates: &[String], haystack: &str) -> bool {
        alternates.iter().any(|v| prefix_dot_plus_fn(&v, haystack))
    }

    let StringMatchOptions { anchor_start, anchor_end, prefix_dot_plus, suffix_dot_plus } = *options;

    match (anchor_start, anchor_end) {
        (true, true) => {
            // ^.+foo.+$
            match (prefix_dot_plus, suffix_dot_plus) {
                (true, true) => {
                    // ^.+foo.+$
                    dot_plus_dot_plus
                }
                (true, false) => {
                    // ^.+foo$
                    dot_plus_ends_with
                }
                (false, true) => {
                    // ^foo.+$
                    starts_with_dot_plus
                }
                _ => {
                    // ^foo$
                    equals
                }
            }
        }
        (true, false) => {
            // ^.+foo
            match (prefix_dot_plus, suffix_dot_plus) {
                (true, true) => {
                    // ^.+foo.+$
                    dot_plus_dot_plus
                }
                (true, false) => {
                    // ^.+foo$
                    dot_plus_ends_with
                }
                (false, true) => {
                    // ^foo.+$
                    starts_with
                }
                _ => {
                    // ^foo$
                    equals
                }
            }
        }
        (false, true) => {
            // foo.+$
            match (prefix_dot_plus, suffix_dot_plus) {
                (true, true) => {
                    // .+foo.+$
                    dot_plus_dot_plus
                }
                (true, false) => {
                    // .+foo$
                    dot_plus_ends_with
                }
                (false, true) => {
                    // foo.+$
                    prefix_dot_plus_
                }
                _ => {
                    // foo$
                    ends_with
                }
            }
        }
        _ => {
            // foo
            match (prefix_dot_plus, suffix_dot_plus) {
                (true, true) => {
                    // .+foo.+
                    dot_plus_dot_plus
                }
                (true, false) => {
                    // .+foo$
                    dot_plus_ends_with
                }
                (false, true) => {
                    // foo.+
                    prefix_dot_plus_
                }
                _ => {
                    // foo
                    contains
                }
            }
        }
    }
}

fn get_literal_match_fn(options: &StringMatchOptions) -> MatchFn {

    let StringMatchOptions { prefix_dot_plus, suffix_dot_plus, anchor_start, anchor_end } = *options;

    // ^foobar.+
    fn start_with_dot_plus_fn(needle: &str, haystack: &str) -> bool {
        haystack.len() > needle.len() && haystack.starts_with(needle)
    }

    match (prefix_dot_plus, suffix_dot_plus) {
        (true, true) => {
            match (anchor_start, anchor_end) {
                (true, true) => {
                    // ^.+foo.+$
                    dot_plus_dot_plus_fn
                }
                (true, false) => {
                    // ^.+foo
                    dot_plus_ends_with_fn
                }
                (false, true) => {
                    // foobar.+$
                    starts_with_fn
                }
                _ => {
                    // .+foobar.+
                    dot_plus_dot_plus_fn
                }
            }
        }
        (true, false) => {
            match (anchor_start, anchor_end) {
                (true, true) => {
                    // ^.+foo.+$
                    dot_plus_dot_plus_fn
                }
                (true, false) => {
                    // ^.+foo
                    dot_plus_contains_fn
                }
                (false, true) => {
                    // .+foo$
                    dot_plus_ends_with_fn
                }
                _ => {
                    // .+foobar
                    dot_plus_contains_fn
                }
            }
        }
        (false, true) => {
            match (anchor_start, anchor_end) {
                (true, true) => {
                    // ^foo.+$
                    start_with_dot_plus_fn
                }
                (true, false) => {
                    // ^foo.+
                    start_with_dot_plus_fn
                }
                (false, true) => {
                    // foo.+$
                    prefix_dot_plus_fn
                }
                _ => {
                    // foo.+
                    prefix_dot_plus_fn
                }
            }
        }
        _ => {
            // match foobar
            match (anchor_start, anchor_end) {
                (true, true) => {
                    // ^foobar$
                    equals_fn
                }
                (true, false) => {
                    // ^foobar
                    starts_with_fn
                }
                (false, true) => {
                    // foobar$
                    ends_with_fn
                }
                _ => {
                    // foobar
                    contains_fn
                }
            }

        }
    }
}

fn equals_fn(needle: &str, haystack: &str) -> bool {
    haystack == needle
}

fn contains_fn(needle: &str, haystack: &str) -> bool {
    haystack.contains(needle)
}

fn starts_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.starts_with(needle)
}

fn ends_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.ends_with(needle)
}

// foobar.+
fn prefix_dot_plus_fn(needle: &str, haystack: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos + needle.len() < haystack.len() - 1
    } else {
        false
    }
}

// ^.+(foo|bar) / .+(foo|bar)
fn dot_plus_contains_fn(needle: &str, haystack: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos > 0
    } else {
        false
    }
}

// ^.+(foo|bar)$ / .+(foo|bar)$
fn dot_plus_ends_with_fn(needle: &str, haystack: &str) -> bool {
    haystack.len() > needle.len() && haystack.ends_with(needle)
}


// ^.+(foo|bar).+
fn dot_plus_dot_plus_fn(needle: &str, haystack: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos > 0 && pos + needle.len() < haystack.len()
    } else {
        false
    }
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