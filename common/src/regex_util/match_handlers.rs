use crate::regex_util::string_pattern::StringPattern;
use get_size::GetSize;
use regex::Regex;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};

const MAX_SET_MATCHES: usize = 256;
/// These cost values are used for sorting tag filters in ascending order or the required CPU
/// time for execution.
///
/// These values are obtained from BenchmarkOptimizedRematch_cost benchmark.
pub const EMPTY_MATCH_COST: usize = 0;
pub const FULL_MATCH_COST: usize = 1;
pub const PREFIX_MATCH_COST: usize = 2;
pub const LITERAL_MATCH_COST: usize = 3;
pub const SUFFIX_MATCH_COST: usize = 4;
pub const MIDDLE_MATCH_COST: usize = 6;
pub const RE_MATCH_COST: usize = 100;
pub const FN_MATCH_COST: usize = 20;

pub type MatchFn = fn(pattern: &str, candidate: &str) -> bool;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Quantifier {
    ZeroOrOne,  // ?
    ZeroOrMore, // *
    OneOrMore,  // +
}

#[derive(Default)]
pub struct StringMatchOptions {
    pub anchor_end: bool,
    pub anchor_start: bool,
    pub prefix_quantifier: Option<Quantifier>,
    pub suffix_quantifier: Option<Quantifier>,
}

impl StringMatchOptions {
    pub fn is_default(&self) -> bool {
        !self.anchor_end
            && !self.anchor_start
            && self.prefix_quantifier.is_none()
            && self.suffix_quantifier.is_none()
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct MatchAnyMatcher {
    match_nl: bool,
}

impl MatchAnyMatcher {
    pub fn new(match_nl: bool) -> Self {
        Self { match_nl }
    }

    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            true
        } else {
            !s.contains('\n')
        }
    }

    fn cost(&self) -> usize {
        FULL_MATCH_COST
    }
}

impl Default for MatchAnyMatcher {
    fn default() -> Self {
        Self { match_nl: true }
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct NonEmptyStringMatcher {
    match_nl: bool,
}

impl NonEmptyStringMatcher {
    pub fn new(match_nl: bool) -> Self {
        Self { match_nl }
    }

    fn matches(&self, s: &str) -> bool {
        if self.match_nl {
            !s.is_empty()
        } else {
            !s.is_empty() && !s.contains('\n')
        }
    }

    fn cost(&self) -> usize {
        FULL_MATCH_COST
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct ZeroOrOneCharsMatcher {
    match_nl: bool,
}

impl ZeroOrOneCharsMatcher {
    fn matches(&self, s: &str) -> bool {
        let empty = s.is_empty();
        if self.match_nl {
            empty || s.chars().count() == 1
        } else {
            empty || (s.chars().count() == 1 && !s.starts_with('\n'))
        }
    }

    fn cost(&self) -> usize {
        FULL_MATCH_COST
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct EqualMultiStringMatcher {
    pub values: Vec<String>,
    is_ascii: bool,
    case_sensitive: bool,
}

impl EqualMultiStringMatcher {
    pub(crate) fn new(case_sensitive: bool, estimated_size: usize) -> Self {
        EqualMultiStringMatcher {
            values: Vec::with_capacity(estimated_size),
            case_sensitive,
            is_ascii: true,
        }
    }

    pub fn push(&mut self, s: String) {
        self.is_ascii = self.is_ascii && s.is_ascii();
        self.values.push(s);
    }

    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    pub fn matches(&self, s: &str) -> bool {
        if self.case_sensitive {
            self.values.iter().any(|v| v == s)
        } else if self.is_ascii {
            self.values.iter().any(|v| v.eq_ignore_ascii_case(s))
        } else {
            let needle = s.to_lowercase();
            self.values.iter().any(|v| v.to_lowercase() == needle)
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn cost(&self) -> usize {
        let match_cost = if self.case_sensitive {
            FULL_MATCH_COST
        } else if self.is_ascii {
            FULL_MATCH_COST * 2
        } else {
            FULL_MATCH_COST * 3
        };
        self.values.len() * match_cost
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralBracketedMatcher {
    pub left: Option<StringPattern>,
    pub matcher: Box<StringMatchHandler>,
    pub right: Option<StringPattern>,
}

impl LiteralBracketedMatcher {
    pub fn new(left: StringPattern, matcher: StringMatchHandler, right: StringPattern) -> Self {
        Self {
            left: Some(left),
            matcher: Box::new(matcher),
            right: Some(right),
        }
    }

    pub fn with_prefix(matcher: StringMatchHandler, prefix: StringPattern) -> Self {
        Self {
            left: Some(prefix),
            matcher: Box::new(matcher),
            right: None,
        }
    }

    pub fn with_suffix(matcher: StringMatchHandler, suffix: StringPattern) -> Self {
        Self {
            left: None,
            matcher: Box::new(matcher),
            right: Some(suffix),
        }
    }

    fn is_case_sensitive(&self) -> bool {
        let is_sensitive = self.matcher.is_case_sensitive();
        match (self.left.as_ref(), self.right.as_ref()) {
            (Some(left), Some(right)) => {
                is_sensitive && left.is_case_sensitive() && right.is_case_sensitive()
            }
            (Some(left), None) => is_sensitive && left.is_case_sensitive(),
            (None, Some(right)) => is_sensitive && right.is_case_sensitive(),
            (None, None) => is_sensitive,
        }
    }

    fn matches(&self, s: &str) -> bool {
        let mut s = s;
        if let Some(left) = &self.left {
            if !left.starts_with(s) {
                return false;
            }
            s = &s[left.len()..];
        }
        if let Some(right) = &self.right {
            if !right.ends_with(s) {
                return false;
            }
            s = &s[..s.len() - right.len()];
        }
        self.matcher.matches(s)
    }

    fn pattern_cost(&self) -> usize {
        let left_cost = self.left.as_ref().map_or(0, |l| l.len());
        let right_cost = self.right.as_ref().map_or(0, |r| r.len());
        left_cost + right_cost
    }

    fn cost(&self) -> usize {
        self.matcher.cost() + self.pattern_cost()
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralMatcher {
    pub left: Option<Box<StringMatchHandler>>,
    pattern: StringPattern,
    pub right: Option<Box<StringMatchHandler>>,
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralPrefixMatcher {
    pub prefix: StringPattern,
    pub right: Option<Box<StringMatchHandler>>,
}

impl LiteralPrefixMatcher {
    pub fn new<S: Into<String>>(
        prefix: S,
        right: Option<Box<StringMatchHandler>>,
        case_sensitive: bool,
    ) -> Self {
        let prefix = StringPattern::new(prefix.into(), case_sensitive);
        Self { prefix, right }
    }

    fn is_case_sensitive(&self) -> bool {
        self.prefix.is_case_sensitive()
    }

    fn cost(&self) -> usize {
        let match_cost = if self.is_case_sensitive() {
            FULL_MATCH_COST
        } else if self.prefix.is_ascii() {
            FULL_MATCH_COST * 2
        } else {
            FULL_MATCH_COST * 2 + 1
        };
        match_cost + self.right.as_ref().map_or(0, |r| r.cost())
    }

    fn matches(&self, s: &str) -> bool {
        if !self.prefix.is_prefix_of(s) {
            return false;
        }

        let Some(suffix) = &self.right else {
            return true;
        };

        let right_part = &s[self.prefix.len()..];
        suffix.matches(right_part)
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralSuffixMatcher {
    pub left: Option<Box<StringMatchHandler>>,
    pub suffix: StringPattern,
}

impl LiteralSuffixMatcher {
    pub fn new<S: Into<String>>(
        left: Option<Box<StringMatchHandler>>,
        suffix: S,
        case_sensitive: bool,
    ) -> Self {
        let suffix = StringPattern::new(suffix.into(), case_sensitive);
        Self { left, suffix }
    }

    fn is_case_sensitive(&self) -> bool {
        self.suffix.is_case_sensitive()
    }

    fn matches(&self, s: &str) -> bool {
        if !self.suffix.is_suffix_of(s) {
            return false;
        }
        let Some(left) = self.left.as_ref() else {
            return true;
        };

        let suffix_len = self.suffix.len();
        if s.len() < suffix_len {
            return false;
        }

        let left_part = &s[..s.len() - suffix_len];
        left.matches(left_part)
    }

    fn cost(&self) -> usize {
        let match_cost = if self.suffix.is_case_sensitive() {
            FULL_MATCH_COST
        } else if self.suffix.is_ascii() {
            FULL_MATCH_COST * 2
        } else {
            FULL_MATCH_COST * 2 + 1
        };
        match_cost + self.left.as_ref().map_or(0, |r| r.cost())
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct ContainsMultiStringMatcher {
    pub substrings: Vec<String>,
    pub left: Option<Box<StringMatchHandler>>,
    pub right: Option<Box<StringMatchHandler>>,
}

impl ContainsMultiStringMatcher {
    pub(crate) fn new(
        substrings: Vec<String>,
        left: Option<StringMatchHandler>,
        right: Option<StringMatchHandler>,
    ) -> Self {
        let left = left.map(Box::new);
        let right = right.map(Box::new);
        Self {
            substrings,
            left,
            right,
        }
    }

    fn matches(&self, s: &str) -> bool {
        for substr in self.substrings.iter() {
            let len = substr.len();
            match (self.left.as_ref(), self.right.as_ref()) {
                (Some(left), Some(right)) => {
                    let mut search_start_pos = 0;
                    while let Some(pos) = s[search_start_pos..].find(substr) {
                        let pos = pos + search_start_pos;
                        let remainder = &s[pos + len..];
                        if left.matches(&s[..pos]) && right.matches(remainder) {
                            return true;
                        }
                        search_start_pos = pos + 1;
                    }
                }
                (Some(left), None) => {
                    if !s.ends_with(substr) {
                        continue;
                    }
                    let left_part = &s[..s.len() - len];
                    if left.matches(left_part) {
                        return true;
                    }
                }
                (None, Some(right)) => {
                    if s.starts_with(substr) && right.matches(&s[len..]) {
                        return true;
                    }
                }
                (None, None) => {
                    if s.contains(substr) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn cost(&self) -> usize {
        let match_cost = MIDDLE_MATCH_COST * self.substrings.len();
        match_cost
            + self.left.as_ref().map_or(0, |l| l.cost())
            + self.right.as_ref().map_or(0, |r| r.cost())
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub struct LiteralMapMatcher {
    pub values: HashSet<String>,
    pub is_case_sensitive: bool,
}

impl LiteralMapMatcher {
    pub(crate) fn new() -> Self {
        Self {
            values: Default::default(),
            is_case_sensitive: true,
        }
    }

    pub(crate) fn push(&mut self, s: String) {
        if self.is_case_sensitive {
            self.values.insert(s.to_lowercase());
        } else {
            self.values.insert(s);
        }
    }

    pub fn set_matches(&self) -> Vec<String> {
        if self.values.len() >= MAX_SET_MATCHES {
            return Vec::new();
        }

        self.values.iter().cloned().collect::<Vec<String>>()
    }

    fn matches(&self, s: &str) -> bool {
        if self.is_case_sensitive {
            self.values.contains(&s.to_lowercase())
        } else {
            self.values.contains(s)
        }
    }
    pub fn is_case_sensitive(&self) -> bool {
        true
    }

    fn cost(&self) -> usize {
        let match_cost = if self.is_case_sensitive {
            FULL_MATCH_COST
        } else {
            FULL_MATCH_COST * 2
        };
        self.values.len() * match_cost
    }
}

#[derive(Debug, Clone, Eq, PartialEq, GetSize)]
pub struct RepetitionMatcher {
    pub sub: String,
    pub min: u32,
    pub max: Option<u32>,
}

impl RepetitionMatcher {
    pub fn new(sub: String, min: u32, max: Option<u32>) -> Self {
        Self { sub, min, max }
    }

    pub fn matches(&self, s: &str) -> bool {
        if self.min == 0 && s.is_empty() {
            return true;
        }
        if self.min == 1 && s == self.sub {
            return true;
        }
        if let Some(max) = self.max {
            let pat_len = self.sub.len();
            let mut cursor = s;
            let mut i = 0;

            while i <= (max + 1) {
                if !cursor.starts_with(&self.sub) {
                    // mismatch at the beginning when min == 0 is handled above
                    return i > self.min;
                }
                i += 1;
                if i > max {
                    return false;
                }
                cursor = &cursor[pat_len..];
                if cursor.len() < pat_len {
                    return i >= self.min;
                }
            }
        }
        true
    }

    fn cost(&self) -> usize {
        let match_cost = LITERAL_MATCH_COST;
        match self.max {
            Some(max) => match_cost * (max as usize + 1),
            None => match_cost * 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegexMatcher {
    pub regex: Regex,
    pub prefix: String,
    pub suffix: String,
    pub set_matches: Vec<String>,
    pub contains: Vec<String>,
    pub string_matcher: Option<Box<StringMatchHandler>>,
}

impl GetSize for RegexMatcher {
    fn get_size(&self) -> usize {
        // TODO: properly calculate a value for the bookkeeping overhead of the regex object
        const REGEX_OVERHEAD: usize = 256;
        REGEX_OVERHEAD
            + self.regex.as_str().get_size()
            + self.prefix.get_size()
            + self.suffix.get_size()
    }
}

impl PartialEq for RegexMatcher {
    fn eq(&self, other: &Self) -> bool {
        self.regex.as_str() == other.regex.as_str()
            && self.prefix == other.prefix
            && self.suffix == other.suffix
            && self.set_matches == other.set_matches
            && self.contains == other.contains
            && self.string_matcher == other.string_matcher
    }
}

impl Eq for RegexMatcher {}

impl RegexMatcher {
    pub(crate) fn new(regex: Regex, prefix: String, suffix: String) -> Self {
        Self {
            regex,
            prefix,
            suffix,
            set_matches: vec![],
            contains: vec![],
            string_matcher: None,
        }
    }

    fn matches(&self, s: &str) -> bool {
        if !self.set_matches.is_empty() && !self.set_matches.iter().any(|x| x.as_str() == s) {
            return false;
        }
        if !self.prefix.is_empty() && !s.starts_with(&self.prefix) {
            return false;
        }
        if !self.suffix.is_empty() && !s.ends_with(&self.suffix) {
            return false;
        }
        if !self.contains.is_empty() && !contains_in_order(s, &self.contains) {
            return false;
        }
        if let Some(ref string_matcher) = self.string_matcher {
            if !string_matcher.matches(s) {
                return false;
            }
        }
        self.regex.is_match(s)
    }

    fn cost(&self) -> usize {
        let match_cost = FULL_MATCH_COST;
        match_cost + self.set_matches.len() * match_cost + self.contains.len() * MIDDLE_MATCH_COST
    }
}

#[derive(Clone, Debug, Eq, PartialEq, GetSize)]
pub struct ConsecutiveLiterals {
    pub(super) prefix: Option<Box<StringMatchHandler>>,
    pub(super) suffix: Option<Box<StringMatchHandler>>,
    pub(super) literals: Vec<StringPattern>,
    _len: usize,
}

impl ConsecutiveLiterals {
    pub fn new(
        prefix: Option<StringMatchHandler>,
        literals: Vec<StringPattern>,
        suffix: Option<StringMatchHandler>,
    ) -> Self {
        let prefix = prefix.map(Box::new);
        let suffix = suffix.map(Box::new);
        let _len = literals.iter().map(|x| x.len()).sum();
        Self {
            literals,
            prefix,
            suffix,
            _len,
        }
    }

    pub fn with_prefix(prefix: StringMatchHandler, literals: Vec<StringPattern>) -> Self {
        let prefix = Box::new(prefix);
        let _len = literals.iter().map(|x| x.len()).sum();
        Self {
            literals,
            prefix: Some(prefix),
            suffix: None,
            _len,
        }
    }

    pub fn matches(&self, s: &str) -> bool {
        let mut cursor = s;
        let pattern_count = self.literals.len();
        let len = self._len;

        if let Some(prefix) = &self.prefix {
            if s.len() < len {
                return false;
            }
            let left_part = &s[..s.len() - len];
            if prefix.matches(left_part) {
                return true;
            }
        }

        for (i, pattern) in self.literals.iter().enumerate() {
            if pattern.starts_with(cursor) {
                let p_len = pattern.len();
                // prevent an out-of-bounds error
                if i < pattern_count && p_len >= cursor.len() {
                    return false;
                }
                cursor = &cursor[p_len..]
            } else {
                return false;
            }
        }
        if let Some(suffix) = &self.suffix {
            return suffix.matches(cursor);
        }
        true
    }

    pub fn cost(&self) -> usize {
        (self.literals.len() * LITERAL_MATCH_COST) + self.suffix.as_ref().map_or(0, |s| s.cost())
    }

    pub fn len(&self) -> usize {
        self._len
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty() && self.prefix.is_none() && self.suffix.is_none()
    }

    pub fn is_case_sensitive(&self) -> bool {
        self.literals.iter().all(|l| l.is_case_sensitive())
    }
}

#[derive(Clone, Debug, GetSize, Eq, PartialEq)]
pub enum StringMatchHandler {
    Alternates(EqualMultiStringMatcher),
    Bracketed(Box<LiteralBracketedMatcher>),
    ConsecutiveLiterals(ConsecutiveLiterals),
    ContainsMulti(ContainsMultiStringMatcher),
    Empty,
    Literal(StringPattern),
    LiteralMap(LiteralMapMatcher),
    MatchAny(MatchAnyMatcher),
    MatchFn(MatchFnHandler),
    MatchNone,
    NotEmpty(NonEmptyStringMatcher),
    Or(Vec<StringMatchHandler>),
    Prefix(LiteralPrefixMatcher),
    Regex(RegexMatcher),
    Repetition(RepetitionMatcher),
    Suffix(LiteralSuffixMatcher),
    ZeroOrOneChars(ZeroOrOneCharsMatcher),
}

impl Default for StringMatchHandler {
    fn default() -> Self {
        Self::MatchAny(MatchAnyMatcher { match_nl: false })
    }
}

impl StringMatchHandler {
    pub fn any(match_nl: bool) -> Self {
        Self::MatchAny(MatchAnyMatcher::new(match_nl))
    }

    #[allow(dead_code)]
    pub fn match_fn(pattern: String, match_fn: MatchFn) -> Self {
        Self::MatchFn(MatchFnHandler::new(pattern, match_fn))
    }

    pub fn empty_string_match() -> Self {
        Self::Empty
    }

    pub fn fast_regex(regex: Regex) -> Self {
        Self::Regex(RegexMatcher::new(regex, String::new(), String::new()))
    }

    pub fn literal_alternates(alts: Vec<String>, case_sensitive: bool) -> Self {
        if alts.len() == 1 {
            let mut alts = alts;
            let pattern = alts.pop().unwrap();
            Self::literal(pattern, case_sensitive)
        } else {
            let mut matcher = EqualMultiStringMatcher::new(true, alts.len());
            matcher.case_sensitive = case_sensitive;
            for alt in alts {
                matcher.push(alt);
            }
            Self::Alternates(matcher)
        }
    }

    pub fn zero_or_one_chars(match_nl: bool) -> Self {
        StringMatchHandler::ZeroOrOneChars(ZeroOrOneCharsMatcher { match_nl })
    }

    pub fn not_empty(match_nl: bool) -> Self {
        StringMatchHandler::NotEmpty(NonEmptyStringMatcher::new(match_nl))
    }

    pub fn literal(value: String, case_sensitive: bool) -> Self {
        let pattern = if case_sensitive {
            StringPattern::case_sensitive(value)
        } else {
            StringPattern::case_insensitive(value)
        };
        StringMatchHandler::Literal(pattern)
    }

    pub fn literal_fn(value: String, options: &StringMatchOptions) -> Self {
        let match_fn = get_literal_match_fn(options);
        StringMatchHandler::MatchFn(MatchFnHandler::new(value, match_fn))
    }

    pub fn equals(value: String) -> Self {
        StringMatchHandler::literal(value, true)
    }

    pub fn prefix(value: String, right: Option<StringMatchHandler>, case_sensitive: bool) -> Self {
        StringMatchHandler::Prefix(LiteralPrefixMatcher::new(
            value,
            right.map(Box::new),
            case_sensitive,
        ))
    }

    pub fn suffix(left: Option<StringMatchHandler>, value: String, case_sensitive: bool) -> Self {
        StringMatchHandler::Suffix(LiteralSuffixMatcher::new(
            left.map(Box::new),
            value,
            case_sensitive,
        ))
    }

    pub fn contains(substring: String) -> Self {
        StringMatchHandler::match_fn(substring, contains_fn)
    }

    pub fn consecutive_literals(
        prefix: Option<StringMatchHandler>,
        literals: Vec<StringPattern>,
        suffix: Option<StringMatchHandler>,
    ) -> Self {
        if literals.len() == 1 && prefix.is_none() && suffix.is_none() {
            let mut literals = literals;
            let pattern = literals.pop().unwrap();
            return StringMatchHandler::Literal(pattern);
        }
        StringMatchHandler::ConsecutiveLiterals(ConsecutiveLiterals::new(prefix, literals, suffix))
    }

    pub fn is_case_sensitive(&self) -> bool {
        match self {
            StringMatchHandler::Alternates(p) => p.is_case_sensitive(),
            StringMatchHandler::Bracketed(p) => p.is_case_sensitive(),
            StringMatchHandler::ConsecutiveLiterals(p) => {
                p.literals.iter().any(|l| l.is_case_sensitive())
            }
            StringMatchHandler::Literal(p) => p.is_case_sensitive(),
            StringMatchHandler::LiteralMap(p) => p.is_case_sensitive(),
            StringMatchHandler::Prefix(p) => p.is_case_sensitive(),
            StringMatchHandler::Suffix(p) => p.is_case_sensitive(),
            _ => true,
        }
    }

    pub fn is_quantifier(&self) -> bool {
        matches!(
            self,
            StringMatchHandler::Repetition(_)
                | StringMatchHandler::ZeroOrOneChars(_)
                | StringMatchHandler::MatchAny(_)
                | StringMatchHandler::MatchNone
        )
    }

    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringMatchHandler::Alternates(m) => m.matches(s),
            StringMatchHandler::Bracketed(p) => p.matches(s),
            StringMatchHandler::ConsecutiveLiterals(m) => m.matches(s),
            StringMatchHandler::ContainsMulti(m) => m.matches(s),
            StringMatchHandler::Empty => s.is_empty(),
            StringMatchHandler::Literal(m) => m.matches(s),
            StringMatchHandler::LiteralMap(m) => m.matches(s),
            StringMatchHandler::MatchAny(m) => m.matches(s),
            StringMatchHandler::MatchFn(m) => m.matches(s),
            StringMatchHandler::MatchNone => false,
            StringMatchHandler::NotEmpty(opts) => opts.matches(s),
            StringMatchHandler::Or(matchers) => matchers.iter().any(|m| m.matches(s)),
            StringMatchHandler::Prefix(m) => m.matches(s),
            StringMatchHandler::Regex(r) => r.matches(s),
            StringMatchHandler::Repetition(m) => m.matches(s),
            StringMatchHandler::Suffix(m) => m.matches(s),
            StringMatchHandler::ZeroOrOneChars(m) => m.matches(s),
        }
    }

    pub fn cost(&self) -> usize {
        match self {
            StringMatchHandler::Alternates(m) => m.cost(),
            StringMatchHandler::Bracketed(m) => m.cost(),
            StringMatchHandler::ConsecutiveLiterals(m) => m.cost(),
            StringMatchHandler::ContainsMulti(m) => m.cost(),
            StringMatchHandler::Empty => EMPTY_MATCH_COST,
            // todo: case-insensitive literals should have a higher cost
            StringMatchHandler::Literal(_) => LITERAL_MATCH_COST,
            StringMatchHandler::LiteralMap(m) => m.cost(),
            StringMatchHandler::MatchAny(m) => m.cost(),
            StringMatchHandler::MatchFn(_) => FN_MATCH_COST,
            StringMatchHandler::MatchNone => EMPTY_MATCH_COST,
            StringMatchHandler::NotEmpty(m) => m.cost(),
            StringMatchHandler::Or(matchers) => matchers.iter().map(|m| m.cost()).sum(),
            StringMatchHandler::Prefix(m) => m.cost(),
            StringMatchHandler::Regex(r) => r.cost(),
            StringMatchHandler::Repetition(m) => m.cost(),
            StringMatchHandler::Suffix(m) => m.cost(),
            StringMatchHandler::ZeroOrOneChars(m) => m.cost(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchFnHandler {
    pattern: String,
    pub(super) match_fn: MatchFn,
}

impl GetSize for MatchFnHandler {
    fn get_size(&self) -> usize {
        self.pattern.get_size() + size_of::<MatchFn>()
    }
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

impl Display for StringMatchHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

pub(super) const fn get_literal_match_fn(options: &StringMatchOptions) -> MatchFn {
    let StringMatchOptions {
        anchor_start,
        anchor_end,
        prefix_quantifier,
        suffix_quantifier,
    } = options;

    // ^foobar.+
    fn start_with_dot_plus_fn(pattern: &str, needle: &str) -> bool {
        needle.len() > pattern.len() && needle.starts_with(pattern)
    }

    // ^.?foo
    fn start_with_zero_or_one_chars_fn(haystack: &str, needle: &str) -> bool {
        if needle == haystack {
            return true;
        }

        if needle.len() == haystack.len() + 1 && needle.ends_with(haystack) {
            return true;
        }

        false
    }

    fn ends_with_zero_or_one_chars_fn(haystack: &str, needle: &str) -> bool {
        if needle == haystack {
            return true;
        }

        if haystack.len() + 1 != needle.len() {
            return false;
        }

        needle.starts_with(haystack)
    }

    // .?foo.?$
    fn zero_or_one_chars_anchors_fn(haystack: &str, needle: &str) -> bool {
        if haystack.len() < needle.len() {
            return false;
        }
        if let Some(pos) = haystack.find(needle) {
            let end = pos + needle.len();
            end == haystack.len() || end == haystack.len() - 1
        } else {
            false
        }
    }

    if *anchor_start && *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // ^.?foo.?$
                zero_or_one_chars_anchors_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrOne)) => {
                // ^.*foo.?$
                contains_dot_quest_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*$
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.*foo.+$
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.+foo.*$
                dot_plus_match_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.+foo.+$
                dot_plus_dot_plus_match_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => {
                // ^.?foo$
                start_with_zero_or_one_chars_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo$
                ends_with_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*$
                starts_with_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // ^.+foo$
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // ^foo.+$
                start_with_dot_plus_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => {
                // ^foo.?$
                ends_with_zero_or_one_chars_fn
            }
            _ => {
                // ^foobar$
                equals_fn
            }
        }
    } else if *anchor_start {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // ^.?foo.?
                start_with_zero_or_one_chars_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.*foo.*
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrOne)) => {
                // ^.*foo.?
                contains_dot_quest_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.*foo.+
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // ^.+foo.*
                dot_plus_match_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // ^.+foo.+
                dot_plus_dot_plus_match_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => start_with_zero_or_one_chars_fn,
            (Some(Quantifier::ZeroOrMore), None) => {
                // ^.*foo
                contains_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => ends_with_zero_or_one_chars_fn,
            (None, Some(Quantifier::ZeroOrMore)) => {
                // ^foo.*
                starts_with_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // ^.+foo
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // ^foo.+
                start_with_dot_plus_fn
            }
            _ => {
                // ^foobar
                starts_with_fn
            }
        }
    } else if *anchor_end {
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // .?foo.?$
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*$
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrOne)) => {
                // .*foo.?$
                contains_dot_quest_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // .*foo.+$
                contains_dot_plus_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .+foo.*$
                dot_plus_match_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // .+foo.+$
                dot_plus_dot_plus_match_fn
            }
            (Some(Quantifier::ZeroOrOne), None) => {
                // .?foo$
                ends_with_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo$
                ends_with_fn
            }
            (None, Some(Quantifier::ZeroOrOne)) => {
                // foo.?$
                ends_with_zero_or_one_chars_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*$
                contains_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // .+foo$
                dot_plus_ends_with_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // foo.+$
                prefix_dot_plus_fn
            }
            _ => {
                // foobar$
                ends_with_fn
            }
        }
    } else {
        // no anchors
        match (prefix_quantifier, suffix_quantifier) {
            (Some(Quantifier::ZeroOrOne), Some(Quantifier::ZeroOrOne)) => {
                // .?foo.?
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .*foo.*
                contains_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::OneOrMore)) => {
                // .*foo.+
                contains_dot_plus_fn
            }
            (Some(Quantifier::ZeroOrMore), Some(Quantifier::ZeroOrOne)) => {
                // .*foo.?
                contains_dot_quest_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::ZeroOrMore)) => {
                // .+foo.*
                dot_plus_match_fn
            }
            (Some(Quantifier::OneOrMore), Some(Quantifier::OneOrMore)) => {
                // .+foo.+
                dot_plus_dot_plus_match_fn
            }
            (Some(Quantifier::ZeroOrMore), None) => {
                // .*foo
                contains_fn
            }
            (None, Some(Quantifier::ZeroOrMore)) => {
                // foo.*
                contains_fn
            }
            (Some(Quantifier::OneOrMore), None) => {
                // .+foo
                dot_plus_match_fn
            }
            (None, Some(Quantifier::OneOrMore)) => {
                // foo.+
                contains_dot_plus_fn
            }
            _ => {
                // foobar
                contains_fn
            }
        }
    }
}

pub(super) fn equals_fn(haystack: &str, needle: &str) -> bool {
    haystack == needle
}

pub(super) fn contains_fn(pattern: &str, needle: &str) -> bool {
    needle.contains(pattern)
}

pub(super) fn starts_with_fn(pattern: &str, needle: &str) -> bool {
    needle.starts_with(pattern)
}

pub(super) fn ends_with_fn(pattern: &str, needle: &str) -> bool {
    needle.ends_with(pattern)
}

// foobar.+
pub(super) fn prefix_dot_plus_fn(haystack: &str, needle: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        pos + needle.len() < haystack.len() - 1
    } else {
        false
    }
}

// .+foobar.*
pub(crate) fn dot_plus_match_fn(pattern: &str, needle: &str) -> bool {
    let mut cursor = &needle[0..];
    while let Some(pos) = cursor.find(pattern) {
        if pos > 0 {
            return true;
        }
        cursor = &cursor[pos + pattern.len()..];
    }
    false
}

// ^.+(foo|bar)$ / .+(foo|bar)$
fn dot_plus_ends_with_fn(pattern: &str, needle: &str) -> bool {
    needle.len() > pattern.len() && needle.ends_with(pattern)
}

// ^.+foo.+
pub(crate) fn dot_plus_dot_plus_match_fn(pattern: &str, needle: &str) -> bool {
    if let Some(pos) = needle.find(pattern) {
        pos > 0 && pos + pattern.len() < needle.len()
    } else {
        false
    }
}

// something like .*foo.+$
pub(super) fn contains_dot_plus_fn(haystack: &str, needle: &str) -> bool {
    if let Some(pos) = haystack.find(needle) {
        let end = pos + needle.len();
        end < haystack.len()
    } else {
        false
    }
}

/// handle  .*aaa.?
pub(super) fn contains_dot_quest_fn(pattern: &str, needle: &str) -> bool {
    let mut needle = needle;
    while let Some(pos) = needle.find(pattern) {
        let remainder = &needle[pos + pattern.len()..];
        if remainder.len() <= 1 {
            return true;
        }
        needle = remainder;
    }
    false
}

pub fn contains_in_order(s: &str, contains: &[String]) -> bool {
    if contains.len() == 1 {
        return s.contains(&contains[0]);
    }
    contains_in_order_multi(s, contains)
}

fn contains_in_order_multi(s: &str, contains: &[String]) -> bool {
    let mut offset = 0;
    for substr in contains {
        if let Some(pos) = s[offset..].find(substr) {
            offset += pos + substr.len();
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
    fn test_zero_or_one_character_string_matcher() {
        // Test case: match newline
        let matcher_match_nl = ZeroOrOneCharsMatcher { match_nl: true };
        assert!(matcher_match_nl.matches(""));
        assert!(matcher_match_nl.matches("x"));
        assert!(matcher_match_nl.matches("\n"));
        assert!(!matcher_match_nl.matches("xx"));
        assert!(!matcher_match_nl.matches("\n\n"));

        // Test case: do not match a newline
        let matcher_no_match_nl = ZeroOrOneCharsMatcher { match_nl: false };
        assert!(matcher_no_match_nl.matches(""));
        assert!(matcher_no_match_nl.matches("x"));
        assert!(!matcher_no_match_nl.matches("\n"));
        assert!(!matcher_no_match_nl.matches("xx"));
        assert!(!matcher_no_match_nl.matches("\n\n"));

        // Test case: Unicode
        let emoji1 = "😀"; // 1 rune
        let emoji2 = "❤️"; // 2 runes
        assert_eq!(emoji1.chars().count(), 1);
        assert_eq!(emoji2.chars().count(), 2);

        let matcher_unicode = ZeroOrOneCharsMatcher { match_nl: true };
        assert!(matcher_unicode.matches(emoji1));
        assert!(!matcher_unicode.matches(emoji2));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, emoji1)));
        assert!(!matcher_unicode.matches(&format!("x{}", emoji1)));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, "x")));
        assert!(!matcher_unicode.matches(&format!("{}{}", emoji1, emoji2)));

        // Test case: invalid Unicode
        let re = Regex::new(r"^.?$").unwrap();
        let matcher_invalid_unicode = ZeroOrOneCharsMatcher { match_nl: true };

        let require_matches = |s: &str, expected: bool| {
            assert_eq!(
                matcher_invalid_unicode.matches(s),
                expected,
                "String: {}",
                s
            );
            assert_eq!(
                re.is_match(s),
                matcher_invalid_unicode.matches(s),
                "String: {}",
                s
            );
        };

        require_matches("\u{FF}", true);
        let value = "x\u{FF}";
        require_matches(value, false);
        require_matches("x\u{FF}x", false);
        require_matches("\u{FF}\u{FE}", false);
    }

    #[test]
    fn test_repetition_matcher_min_zero_empty_string() {
        let matcher = RepetitionMatcher::new("sub".to_string(), 0, Some(3));
        assert!(matcher.matches(""));
    }

    #[test]
    fn test_repetition_exact_match() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 1, None);
        assert!(matcher.matches("abc"));
    }

    #[test]
    fn test_repetition_min_zero() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 0, None);
        assert!(matcher.matches(""));
        assert!(matcher.matches("abc"));
        assert!(matcher.matches("abcabc"));
    }

    #[test]
    fn test_repetition_zero_to_n() {
        let matcher = RepetitionMatcher::new("abc".to_string(), 0, Some(3));
        assert!(matcher.matches(""));
        assert!(matcher.matches("abc"));
        assert!(matcher.matches("abcabc"));
        assert!(matcher.matches("abcabcabc"));

        assert_eq!(false, matcher.matches("bbb"));
        assert_eq!(false, matcher.matches("abcabcabcabc"));
    }
}
