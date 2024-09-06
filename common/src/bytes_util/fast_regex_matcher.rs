use std::fmt::{Debug, Display};

use dynamic_lru_cache::DynamicCache;
use regex::Regex;

const DEFAULT_CACHE_SIZE: usize = 100;

/// FastRegexMatcher implements fast matcher for strings.
///
/// It caches string match results and returns them back on the next calls
/// without calling the match_func, which may be expensive.
pub struct FastRegexMatcher {
    regex: Regex,
    cache: DynamicCache<String, bool>,
}

impl FastRegexMatcher {
    /// creates new matcher which applies match_func to strings passed to matches()
    ///
    /// match_func must return the same result for the same input.
    #[allow(unused)]
    pub fn new(regex: Regex) -> Self {
        Self {
            regex,
            cache: DynamicCache::new(DEFAULT_CACHE_SIZE),
        }
    }

    /// Applies match_func to s and returns the result.
    pub fn matches(&self, s: &str) -> bool {
        let key = s.to_string();
        self.matches_string(&key)
    }

    pub fn matches_string(&self, s: &String) -> bool {
        let res = self.cache.get_or_insert(s, || self.regex.is_match(s));
        *res
    }
}

impl Clone for FastRegexMatcher {
    fn clone(&self) -> Self {
        let cache = self.cache.clone();
        Self {
            regex: self.regex.clone(),
            cache,
        }
    }
}

impl PartialEq for FastRegexMatcher {
    fn eq(&self, other: &Self) -> bool {
        self.regex.as_str() == other.regex.as_str()
    }
}

impl Display for FastRegexMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "/{}/", self.regex)
    }
}

impl Debug for FastRegexMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "/{}/", self.regex)
    }
}