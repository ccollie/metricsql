use crate::bytes_util::string_transform_cacher::StringTransformCache;

/// FastStringTransformer implements fast transformer for strings.
///
/// It caches transformed strings and returns them back on the next calls
/// without calling the transformFunc, which may be expensive.
pub struct FastStringTransformer {
    inner: StringTransformCache<String>,
}

impl FastStringTransformer {
    /// creates new transformer, which applies transform_func to strings passed to transform()
    ///
    /// transform_func must return the same result for the same input.
    pub fn new(transform_func: fn(s: &str) -> String) -> FastStringTransformer {
        FastStringTransformer {
            inner: StringTransformCache::new(transform_func),
        }
    }

    /// transform applies transformFunc to s and returns the result.
    pub fn transform(&self, s: &str) -> String {
        self.inner.transform(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_string_transformer() {
        fn check(s: &str, expected: &str) {
            let fst = FastStringTransformer::new(|s| s.to_uppercase());
            for i in 0..10 {
                let result = fst.transform(s);
                assert_eq!(result, expected,
                           "unexpected result for transform({s}) at iteration {i}; got {result}; want {expected}")
            }
        }

        check("a", "A");
        check("a", "A");
        check("b", "B");
        check("", "");
        check("foo", "FOO");
        check("a_b-C", "A_B-C")
    }
}
