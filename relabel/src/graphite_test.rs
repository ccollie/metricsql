#[cfg(test)]
mod tests {
    use crate::{GraphiteMatchTemplate, GraphiteReplaceTemplate};

    #[test]
    fn test_graphite_template_match_expand() {
        fn check(match_tpl: &str, s: &str, replace_tpl: &str, result_expected: &str) {
            let gmt = GraphiteMatchTemplate::new(match_tpl);
            let mut matches: Vec<String> = vec![];
            let ok = gmt.is_match(&mut matches, s);

            let grt = GraphiteReplaceTemplate::new(replace_tpl);
            let result = grt.expand(&matches);
            assert_eq!(result, result_expected, "unexpected result; got {}; want {}", result, result_expected);
        }

        check("", "", "", "");
        check("test.*.*.counter", "test.foo.bar.counter", "${2}_total", "bar_total");
        check("test.*.*.counter", "test.foo.bar.counter", "$1_total", "foo_total");
        check("test.*.*.counter", "test.foo.bar.counter", "total_$0", "total_test.foo.bar.counter");
        check("test.dispatcher.*.*.*", "test.dispatcher.foo.bar.baz", "$3-$2-$1", "baz-bar-foo");
        check("*.signup.*.*", "foo.signup.bar.baz", "$1-${3}_$2_total", "foo-baz_bar_total")
    }

    #[test]
    fn test_graphite_match_template_match() {
        fn check(tpl: &str, s: &str, matches_expected: Vec<&str>, ok_expected: bool) {
            let gmt = GraphiteMatchTemplate::new(tpl);
            let tpl_got = gmt.to_string();
            assert_eq!(tpl_got, tpl, "unexpected template; got {}; want {}", tpl_got, tpl);
            let mut matches = vec![];
            let ok = gmt.is_match(&mut matches, s);
            assert_eq!(ok, ok_expected,
                       "unexpected ok result for tpl={tpl}, s={s}; got {ok}; want {ok_expected}");
            if ok_expected {
                assert_eq!(matches, matches_expected,
                           "unexpected matches for tpl={tpl}, s={s}; got\n{:?}\nwant\n{:?}\ngraphiteMatchTemplate={gmt}",
                           matches, matches_expected);
            }
        }

        check("", "", vec![""], true);
        check("", "foobar", vec![], false);
        check("foo", "foo", vec!["foo"], true);
        check("foo", "",  vec![], false);
        check("foo.bar.baz", "foo.bar.baz", vec!["foo.bar.baz"], true);
        check("*", "foobar", vec!["foobar", "foobar"], true);
        check("**", "foobar", vec![], false);
        check("*", "foo.bar", vec![], false);
        check("*foo", "barfoo", vec!["barfoo", "bar"], true);
        check("*foo", "foo", vec!["foo", ""], true);
        check("*foo", "bar.foo", vec![], false);
        check("foo*", "foobar", vec!["foobar", "bar"], true);
        check("foo*", "foo", vec!["foo", ""], true);
        check("foo*", "foo.bar", vec![], false);
        check("foo.*", "foobar", vec![], false);
        check("foo.*", "foo.bar", vec!["foo.bar", "bar"], true);
        check("foo.*", "foo.bar.baz", vec![], false);
        check("*.*.baz", "foo.bar.baz", vec!["foo.bar.baz", "foo", "bar"], true);
        check("*.bar", "foo.bar.baz", vec![], false);
        check("*.bar", "foo.baz", vec![], false)
    }

    #[test]
    fn test_graphite_replace_template_expand() {
        fn check(tpl: &str, matches: Vec<&str>, result_expected: &str) {
            let grt = GraphiteReplaceTemplate::new(tpl);
            let tpl_got = grt.to_string();
            assert_eq!(tpl_got, tpl, "unexpected template; got {}; want {}", tpl_got, tpl);
            let to_expand = matches.iter().map(|s| s.to_string()).collect();
            let result = grt.expand(&to_expand);
            assert_eq!(result, result_expected, "unexpected result for tpl={}; got {}; want {}", tpl, result, result_expected);
        }

        check("", vec![], "");
        check("foo", vec![], "foo");
        check("$", vec![], "$");
        check("$1", vec![], "$1");
        check("${123", vec![], "${123");
        check("${123}", vec![], "${123}");
        check("${foo}45$sdf$3", vec![], "${foo}45$sdf$3");
        check("$1", vec!["foo", "bar"], "bar");
        check("$0-$1", vec!["foo", "bar"], "foo-bar");
        check("x-${0}-$1", vec!["foo", "bar"], "x-foo-bar")
    }
}