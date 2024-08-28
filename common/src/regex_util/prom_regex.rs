use predicates::reflection::PredicateReflection;
use predicates::Predicate;
use regex::Error as RegexError;
use std::fmt::{Display, Formatter};

use super::match_handlers::StringMatchHandler;
use super::regex_utils::{get_prefix_matcher, get_suffix_matcher, simplify};

/// PromRegex implements an optimized string matching for Prometheus-like regex.
///
/// The following regexes are optimized:
///
/// - plain string such as "foobar"
/// - alternate strings such as "foo|bar|baz"
/// - prefix match such as "foo.*" or "foo.+"
/// - substring match such as ".*foo.*" or ".+bar.+"
///
/// The rest of regexes are also optimized by returning cached match results for the same input strings.
#[derive(Clone, Debug)]
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

impl PredicateReflection for PromRegex {}

impl Display for PromRegex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PromRegex({}, {})",
            self.prefix_matcher, self.suffix_matcher
        )
    }
}

impl Predicate<&str> for PromRegex {
    fn eval(&self, variable: &&str) -> bool {
        self.match_string(variable)
    }
}

#[cfg(test)]
mod test {
    use regex::Regex;

    use super::PromRegex;

    #[test]
    fn test_prom_regex_parse_failure() {
        fn f(expr: &str) {
            let _ = PromRegex::new(expr).expect("unexpected success for expr={expr}");
        }

        f("fo[bar");
        f("foo(bar")
    }

    #[test]
    fn test_prom_regex() {
        fn f(expr: &str, s: &str, result_expected: bool) {
            let pr = PromRegex::new(expr).expect("unexpected failure");
            let result = pr.match_string(s);
            assert_eq!(
                result, result_expected,
                "unexpected result when matching \"{expr}\" against \"{s}\"; got {result}; want {result_expected}"
            );

            // Make sure the result is the same for regular regexp
            let expr_anchored = "^(?:".to_owned() + expr + ")$";
            let re = Regex::new(&expr_anchored).expect("unexpected failure");
            let result = re.is_match(s);
            assert_eq!(
                result, result_expected,
                "unexpected result when matching {expr_anchored} against {s}; got {result}; want {result_expected}"
            );
        }

        f("^foo|b(ar)$", "foo", true);

        f("", "foo", false);
        f("", "", true);
        f("", "foo", false);
        f("foo", "", false);
        f(".*", "", true);
        f(".*", "foo", true);
        f(".+", "", false);
        f(".+", "foo", true);
        f("foo.*", "bar", false);
        f("foo.*", "foo", true);
        f("foo.*", "foobar", true);
        f("foo.+", "bar", false);
        f("foo.+", "foo", false);
        f("foo.+", "foobar", true);
        f("foo|bar", "", false);
        f("foo|bar", "a", false);
        f("foo|bar", "foo", true);
        f("foo|bar", "bar", true);
        f("foo|bar", "foobar", false);
        f("foo(bar|baz)", "a", false);
        f("foo(bar|baz)", "foobar", true);
        f("foo(bar|baz)", "foobaz", true);
        f("foo(bar|baz)", "foobaza", false);
        f("foo(bar|baz)", "foobal", false);
        f("^foo|b(ar)$", "foo", true);
        f("^foo|b(ar)$", "bar", true);
        f("^foo|b(ar)$", "ar", false);
        f(".*foo.*", "foo", true);
        f(".*foo.*", "afoobar", true);
        f(".*foo.*", "abc", false);
        f("foo.*bar.*", "foobar", true);
        f("foo.*bar.*", "foo_bar_", true);
        f("foo.*bar.*", "foobaz", false);
        f(".+foo.+", "foo", false);
        f(".+foo.+", "afoobar", true);
        f(".+foo.+", "afoo", false);
        f(".+foo.+", "abc", false);
        f("foo.+bar.+", "foobar", false);
        f("foo.+bar.+", "foo_bar_", true);
        f("foo.+bar.+", "foobaz", false);
        f(".+foo.*", "foo", false);
        f(".+foo.*", "afoo", true);
        f(".+foo.*", "afoobar", true);
        f(".*(a|b).*", "a", true);
        f(".*(a|b).*", "ax", true);
        f(".*(a|b).*", "xa", true);
        f(".*(a|b).*", "xay", true);
        f(".*(a|b).*", "xzy", false);
        f("^(?:true)$", "true", true);
        f("^(?:true)$", "false", false)
    }
}
