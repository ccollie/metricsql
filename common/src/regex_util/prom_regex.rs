use regex::Error as RegexError;
use super::simplify::simplify;
use super::get_optimized_re_match_func;
use super::StringMatchHandler;

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
#[derive(Clone, Debug)]
pub struct PromRegex {
    /// prefix contains literal prefix for regex.
    /// For example, prefix="foo" for regex="foo(a|b)"
    pub prefix: String,
    pub matcher: StringMatchHandler,
    pub is_complete: bool,
}

impl Default for PromRegex {
    fn default() -> Self {
        Self::new(".*").unwrap()
    }
}

impl PromRegex {
    pub fn new(expr: &str) -> Result<PromRegex, RegexError> {
        let (prefix, suffix) = simplify(expr)?;
        let (matcher, _) = get_optimized_re_match_func(expr)?;
        let pr = PromRegex {
            prefix,
            matcher,
            is_complete: suffix.is_empty(),
        };
        Ok(pr)
    }

    /// is_match returns true if s matches pr.
    ///
    /// The pr is automatically anchored to the beginning and to the end
    /// of the matching string with '^' and '$'.
    pub fn is_match(&self, s: &str) -> bool {
        self.matcher.matches(s)
    }
}