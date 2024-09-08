use super::string_transform_cacher::StringTransformCache;
use std::fmt::{Display, Formatter};

/// FastStringMatcher implements fast matcher for strings.
///
/// It caches string match results and returns them back on the next calls
/// without calling the match_func, which may be expensive.
#[derive(Clone, Debug)]
pub struct FastStringMatcher {
    inner: StringTransformCache<bool>,
}

impl FastStringMatcher {
    /// creates new matcher which applies match_func to strings passed to matches()
    ///
    /// match_func must return the same result for the same input.
    pub fn new(match_func: fn(s: &str) -> bool) -> Self {
        Self {
            inner: StringTransformCache::new(match_func),
        }
    }

    // Match applies match_func to s and returns the result.
    pub fn matches(&self, s: &str) -> bool {
        self.inner.transform(s)
    }
}

impl Display for FastStringMatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FastStringMatcher(value)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_string_matcher() {
        fn f(s: &str, expected: bool) {
            let fsm = FastStringMatcher::new(|s: &str| -> bool { s.starts_with("foo") });
            for i in 0..10 {
                let result = fsm.matches(s);
                assert_eq!(result, expected,
                           "unexpected result for matches({s}) at iteration {i}; got {result}; want {expected}");
            }
        }

        f("", false);
        f("foo", true);
        f("a_b-C", false);
        f("foobar", true)
    }
}
