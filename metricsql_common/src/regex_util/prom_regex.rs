use regex::Error as RegexError;

use crate::regex_util::match_handlers::StringMatchHandler;
use crate::regex_util::regex_utils::{get_prefix_matcher, get_suffix_matcher, simplify};

/// PromRegex implements an optimized string matching for Prometheus-like regex.
///
/// The following regexes are optimized:
///
/// - plain string such as "foobar"
/// - alternate strings such as "foo|bar|baz"
/// - prefix match such as "foo.*" or "foo.+"
/// - substring match such as ".*foo.*" or ".+bar.+"
///
/// The rest of regexps are also optimized by returning cached match results for the same input strings.
pub struct PromRegex {
    /// prefix contains literal prefix for regex.
    /// For example, prefix="foo" for regex="foo(a|b)"
    prefix: String,
    prefix_matcher: StringMatchHandler,
    suffix_matcher: StringMatchHandler,
}

impl Default for PromRegex {
    fn default() -> Self {
        Self::new(".*").unwrap()
    }
}

impl PromRegex {
    pub fn new(expr: &str) -> Result<PromRegex, RegexError> {
        let (prefix, suffix) = simplify(expr)?;
        let pr = PromRegex {
            prefix: prefix.to_string(),
            prefix_matcher: get_prefix_matcher(&prefix),
            suffix_matcher: get_suffix_matcher(&suffix)?,
        };
        Ok(pr)
    }

    /// match_string returns true if s matches pr.
    ///
    /// The pr is automatically anchored to the beginning and to the end
    /// of the matching string with '^' and '$'.
    pub fn match_string(&self, s: &str) -> bool {
        if self.prefix_matcher.matches(s) {
            if let Some(suffix) = s.strip_prefix(&self.prefix) {
                return self.suffix_matcher.matches(suffix);
            }
        }
        false
    }
}
