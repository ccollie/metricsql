#[cfg(test)]
mod tests {
    use crate::common::regex_util::tag_filter::{get_regexp_from_cache, TagFilter, TagFilters};

    #[test]
    fn test_get_regexp_from_cache() {
        fn f(s: &str, expected_matches: &[&str], expected_mismatches: &[&str], suffix_expected: &str) {
            for _ in 0..3 {
                let rcv = match get_regexp_from_cache(s) {
                    Ok(rcv) => rcv.clone(),
                    Err(err) => {
                        panic!("cannot get regexp from cache for s={s}: {:?}", err);
                    }
                };
                let matcher = &rcv.re_match;
                for expected_match in expected_matches.iter() {
                    assert!(matcher.matches(expected_match), "s={:?} must match {expected_match}", s);
                }
                for expected_mismatch in expected_mismatches {
                    assert!(!matcher.matches(expected_mismatch), "s={:?} must mismatch {:?}", s, expected_mismatches);
                }
            }
        }

       // f("",  &[""], &["foo", "x"], "");
       // f("foo", &["foo"], &["", "bar"], "");
     //   f("(?s)(foo)?", &["foo", ""], &["s", "bar"], "");
        f("foo.*", &["foo", "foobar"], &["xfoo", "xfoobar", "", "a"], "");
      //  f("foo(a|b)?", &[], &["fooa", "foob", "foo"], &["xfoo", "xfoobar", "", "fooc", "fooba"], "");
        f(".*foo", &["foo", "xfoo"], &["foox", "xfoobar", "", "a"], "foo");
      //  f("(a|b)?foo", &[], &["foo", "afoo", "bfoo"], &["foox", "xfoobar", "", "a"], "foo");
        f(".*foo.*",  &["foo", "xfoo", "foox", "xfoobar"], &["", "bar", "foxx"], "");
        f(".*foo.+",  &["foo1", "xfoodff", "foox", "xfoobar"], &["", "bar", "foo", "fox"], "");
        f(".+foo.+",  &["xfoo1", "xfoodff", "xfoox", "xfoobar"], &["", "bar", "foo", "foox", "xfoo"], "");
        f(".+foo.*", &["xfoo", "xfoox", "xfoobar"], &["", "bar", "foo", "fox"], "");
     // f(".+foo(a|b)?", &[], &["xfoo", "xfooa", "xafoob"], &["", "bar", "foo", "foob"], "");
     //   f(".*foo(a|b)?", &[], &["foo", "foob", "xafoo", "xfooa"], &["", "bar", "fooba"], "");
    //    f("(a|b)?foo(a|b)?", &[], &["foo", "foob", "afoo", "afooa"], &["", "bar", "fooba", "xfoo"], "");
        f("((.*)foo(.*))", &["foo", "xfoo", "foox", "xfoobar"], &["", "bar", "foxx"], "");
        f(".+foo", &["afoo", "bbfoo"], &["foo", "foobar", "afoox", ""], "foo");
        f("a|b", &["a", "b"], &["xa", "bx", "xab", ""], "");
        f("(a|b)", &["a", "b"], &["xa", "bx", "xab", ""], "");
        f("(a|b)foo(c|d)",  &["afooc", "bfood"], &["foo", "", "afoo", "fooc", "xfood"], "");
        f("foo.+", &["foox", "foobar"], &["foo", "afoox", "afoo", ""], "");
        f(".*foo.*bar", &["foobar", "xfoobar", "xfooxbar", "fooxbar"], &["", "foobarx", "afoobarx", "aaa"], "bar");
        f("foo.*bar", &["foobar", "fooxbar"], &["xfoobar", "", "foobarx", "aaa"], "bar");
        f("foo.*bar.*", &["foobar", "fooxbar", "foobarx", "fooxbarx"], &["", "afoobarx", "aaa", "afoobar"], "");
        f("foo.*bar.*baz", &["foobarbaz", "fooxbarxbaz", "foobarxbaz", "fooxbarbaz"], &["", "afoobarx", "aaa", "afoobar", "foobarzaz"], "baz");
        f(".+foo.+(b|c).+",  &["xfooxbar", "xfooxca"], &["", "foo", "foob", "xfooc", "xfoodc"], "");

        f("(?i)foo", &["foo", "Foo", "FOO"], &["xfoo", "foobar", "xFOObar"], "");
        f("(?i).+foo", &["xfoo", "aaFoo", "bArFOO"], &["foosdf", "xFOObar"], "");
        f("(?i)(foo|bar)", &["foo", "Foo", "BAR", "bAR"], &["foobar", "xfoo", "xFOObAR"], "");
        f("(?i)foo.*bar", &["foobar", "FooBAR", "FOOxxbaR"], &["xfoobar", "foobarx", "xFOObarx"], "");

        f(".*", &["", "a", "foo", "foobar"], &[], "");
        f("foo|.*", &["", "a", "foo", "foobar"], &[], "");
        f(".+", &["a", "foo"], &[""], "");
        f("(.+)*(foo)?", &["a", "foo", ""], &[], "");

        // Graphite-like regexps
        f(r#"foo\.[^.]*\.bar\.ba(xx|zz)[^.]*\.a"#, &["foo.ss.bar.baxx.a", "foo.s.bar.bazzasd.a"], &["", "foo", "foo.ss.xar.baxx.a"], ".a");
        f(r#"foo\.[^.]*?\.bar\.baz\.aaa"#, &["foo.aa.bar.baz.aaa"], &["", "foo"], ".bar.baz.aaa");
    }

    fn mismatches_suffix(tf: &TagFilter, suffix: &str) {
        let ok = tf.matches(suffix);
        if ok != tf.is_negative {
            panic!("{} mustn't match suffix {}", tf, suffix)
        }
    }

    fn matches_suffix(tf: &TagFilter, suffix: &str) {
        let ok = tf.matches(suffix);
        if ok == tf.is_negative {
            panic!("{} must match suffix {}", tf, suffix)
        }
    }

    fn init_tf(value: &str, is_negative: bool, is_regexp: bool) -> TagFilter {
        let key = "key";
        let tf: TagFilter = TagFilter::new(key, value, is_negative, is_regexp).unwrap();
        tf
    }

    fn tv_no_trailing_tag_separator(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn plain_value() {
        let value = "xx";
        let is_negative = false;
        let is_regexp = false;
        let expected_prefix = tv_no_trailing_tag_separator(value);
        let tf = init_tf(value, is_negative, is_regexp);

        // Plain value must match empty suffix only
        matches_suffix(&tf, "");
        mismatches_suffix(&tf, "foo");
        mismatches_suffix(&tf, "xx")
    }

    #[test]
    fn negative_plain_value() {
        let value = "xx";
        let is_negative = true;
        let is_regexp = false;
        let expected_prefix = tv_no_trailing_tag_separator(value);
        let tf = init_tf(value, is_negative, is_regexp);

        // Negative plain value must match all except empty suffix
        mismatches_suffix(&tf, "");
        matches_suffix(&tf, "foo");
        matches_suffix(&tf, "foxx");
        matches_suffix(&tf, "xx");
        matches_suffix(&tf, "xxx");
        matches_suffix(&tf, "xxfoo");
    }

    #[test]
    fn regexp_convert_to_plain_value() {
        let value = "http";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match only empty suffix
        matches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        mismatches_suffix(&tf, "http");
        mismatches_suffix(&tf, "foobar")
    }

    #[test]
    fn negative_regexp_convert_to_plain_value() {
        let value = "http";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match all except empty suffix
        mismatches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "xhttp");
        matches_suffix(&tf, "http");
        matches_suffix(&tf, "httpx");
        matches_suffix(&tf, "foobar");
    }

    #[test]
    fn regexp_prefix_any_suffix() {
        let value = "http.*";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match any suffix
        matches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "http");
        matches_suffix(&tf, "foobar");
    }

    #[test]
    fn negative_regexp_prefix_any_suffix() {
        let value = "http.*";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Mustn't match any suffix
        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        mismatches_suffix(&tf, "xhttp");
        mismatches_suffix(&tf, "foobar");
    }

    #[test]
    fn regexp_prefix_contains_suffix() {
        let value = "http.*foo.*";
        let is_negative = false;
        let is_regexp = true;
        let expected_prefix = tv_no_trailing_tag_separator("http");
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match any suffix with `foo`
        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        mismatches_suffix(&tf, "http");
        matches_suffix(&tf, "foo");
        matches_suffix(&tf, "foobar");
        matches_suffix(&tf, "xfoobar");
        matches_suffix(&tf, "xfoo");
    }

    #[test]
    fn negative_regexp_prefix_contains_suffix() {
        let value = "http.*foo.*";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match any suffix without `foo`
        matches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "http");
        mismatches_suffix(&tf, "foo");
        mismatches_suffix(&tf, "foobar");
        mismatches_suffix(&tf, "xfoobar");
        mismatches_suffix(&tf, "xfoo");
        mismatches_suffix(&tf, "httpfoo");
        mismatches_suffix(&tf, "httpfoobar");
        mismatches_suffix(&tf, "httpxfoobar");
        mismatches_suffix(&tf, "httpxfoo");
    }

    #[test]
    fn negative_regexp_noprefix_contains_suffix() {
        let value = ".*foo.*";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match anything not matching `.*foo.*`
        matches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "http");
        mismatches_suffix(&tf, "foo");
        mismatches_suffix(&tf, "foobar");
        mismatches_suffix(&tf, "xfoobar");
        mismatches_suffix(&tf, "xfoo");
    }

    #[test]
    fn regexp_prefix_special_suffix() {
        let value = "http.*bar";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match suffix ending on bar
        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        matches_suffix(&tf, "bar");
        mismatches_suffix(&tf, "barx");
        matches_suffix(&tf, "foobar");
        mismatches_suffix(&tf, "foobarx");
    }

    #[test]
    fn negative_regexp_prefix_special_suffix() {
        let value = "http.*bar";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Mustn't match suffix ending on bar
        matches_suffix(&tf, "");
        mismatches_suffix(&tf, "bar");
        mismatches_suffix(&tf, "xhttpbar");
        mismatches_suffix(&tf, "httpbar");
        matches_suffix(&tf, "httpbarx");
        mismatches_suffix(&tf, "httpxybar");
        matches_suffix(&tf, "httpxybarx");
        mismatches_suffix(&tf, "ahttpxybar");
    }

    #[test]
    fn negative_regexp_noprefix_special_suffix() {
        let value = ".*bar";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match all except the regexp from value
        matches_suffix(&tf, "");
        mismatches_suffix(&tf, "bar");
        mismatches_suffix(&tf, "xhttpbar");
        matches_suffix(&tf, "barx");
        matches_suffix(&tf, "pbarx");
    }

    #[test]
    fn regexp_or_suffixes() {
        let value = "http(foo|bar)";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match foo or bar suffix
        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        matches_suffix(&tf, "bar");
        mismatches_suffix(&tf, "barx");
        matches_suffix(&tf, "foo");
        mismatches_suffix(&tf, "foobar");
    }

    #[test]
    fn negative_regexp_or_suffixes() {
        let value = "http(foo|bar)";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Mustn't match foo or bar suffix
        matches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        mismatches_suffix(&tf, "foo");
        matches_suffix(&tf, "fooa");
        matches_suffix(&tf, "xfooa");
        mismatches_suffix(&tf, "bar");
        matches_suffix(&tf, "xhttpbar");
    }

    #[test]
    fn regexp_iflag_no_suffix() {
        let value = "(?i)http";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match case-insenstive http
        matches_suffix(&tf, "http");
        matches_suffix(&tf, "HTTP");
        matches_suffix(&tf, "hTTp");

        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "foobar");
        mismatches_suffix(&tf, "xhttp");
        mismatches_suffix(&tf, "xhttp://");
        mismatches_suffix(&tf, "hTTp://foobar.com");
    }

    #[test]
    fn negative_regexp_iflag_no_suffix() {
        let value = "(?i)http";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Mustn't match case-insensitive http
        mismatches_suffix(&tf, "http");
        mismatches_suffix(&tf, "HTTP");
        mismatches_suffix(&tf, "hTTp");

        matches_suffix(&tf, "");
        matches_suffix(&tf, "foobar");
        matches_suffix(&tf, "xhttp");
        matches_suffix(&tf, "xhttp://");
        matches_suffix(&tf, "hTTp://foobar.com");
    }

    #[test]
    fn regexp_iflag_any_suffix() {
        let value = "(?i)http.*";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Must match case-insensitive http
        matches_suffix(&tf, "http");
        matches_suffix(&tf, "HTTP");
        matches_suffix(&tf, "hTTp://foobar.com");

        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "foobar");
        mismatches_suffix(&tf, "xhttp");
        mismatches_suffix(&tf, "xhttp://");
    }

    #[test]
    fn negative_regexp_iflag_any_suffix() {
        let value = "(?i)http.*";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        // Mustn't match case-insensitive http
        mismatches_suffix(&tf, "http");
        mismatches_suffix(&tf, "HTTP");
        mismatches_suffix(&tf, "hTTp://foobar.com");

        matches_suffix(&tf, "");
        matches_suffix(&tf, "foobar");
        matches_suffix(&tf, "xhttp");
        matches_suffix(&tf, "xhttp://");
    }

    #[test]
    fn non_empty_string_regexp_negative_matches() {
        let value = ".+";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);
        matches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        mismatches_suffix(&tf, "foo");
    }

    #[test]
    fn non_empty_string_regexp_matches() {
        let value = ".+";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        mismatches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "foo");
    }

    #[test]
    fn match_all_regexp_negative_matches() {
        let value = ".*";
        let is_negative = true;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        mismatches_suffix(&tf, "");
        mismatches_suffix(&tf, "x");
        mismatches_suffix(&tf, "foo");
    }

    #[test]
    fn match_all_regexp_matches() {
        let value = ".*";
        let is_negative = false;
        let is_regexp = true;
        let tf = init_tf(value, is_negative, is_regexp);

        matches_suffix(&tf, "");
        matches_suffix(&tf, "x");
        matches_suffix(&tf, "foo");
    }

    #[test]
    fn test_tag_filters_string() {
        let mut tfs = Default::default();
        let must_add = |tfs: &mut TagFilters, key: &str, value: &str, is_negative: bool, is_regexp: bool| {
            tfs.add(key, value, is_negative, is_regexp).unwrap()
        };

        must_add(&mut tfs, "", "metric_name", false, false);
        must_add(&mut tfs, "tag_re", "re.value", false, true);
        must_add(&mut tfs, "tag_nre", "nre.value", true, true);
        must_add(&mut tfs, "tag_n", "n_value", true, false);
        must_add(&mut tfs, "tag_re_graphite", "foo\\.bar", false, true);

        let s = tfs.to_string();
        let expected = r#"{__name__="metric_name",tag_re=~"re.value",tag_nre!~"nre.value",tag_n!="n_value",tag_re_graphite="foo.bar"]"#;
        assert_eq!(s, expected, "unexpected TagFilters.to_string(); got {s}; want {expected}")
    }

    #[test]
    fn test_tag_filters_add_empty() {
        let tfs = &mut Default::default();

        let must_add = |tfs: &mut TagFilters, key: &str, value: &str, is_negative: bool, is_regexp: bool| {
            tfs.add(key, value, is_negative, is_regexp).unwrap()
        };

        let expect_tag_filter = |tfs: &TagFilters, idx: usize, value: &str, is_negative: bool, is_regexp: bool| {
            if idx >= tfs.len() {
                panic!("missing tag filter #{}; len(tfs)={}, tfs={:?}", idx, tfs.len(), tfs)
            }
            let tf = tfs.get(idx).unwrap();
            assert_eq!(tf.value, value, "unexpected tag filter value; got {}; want {value}", tf.value);
            assert_eq!(tf.is_negative, is_negative, "unexpected tag filter is_negative; got {}; want {is_negative}", tf.is_negative);
            assert_eq!(tf.is_regexp, is_regexp,
                       "unexpected is_regexp; got {}; want {}", tf.is_regexp, is_regexp)
        };

        // Empty filters
        must_add(tfs, "", "", false, false);
        expect_tag_filter(tfs, 0, ".+", true, true);
        must_add(tfs, "foo", "", false, false);
        expect_tag_filter(&tfs, 1, ".+", true, true);
        must_add(tfs, "foo", "", true, false);
        expect_tag_filter(&tfs, 2, ".+", false, true);

        // Empty regexp filters
        tfs.reset();
        must_add(tfs, "foo", ".*", false, true);
        if tfs.len() != 0 {
            panic!("unexpectedly added empty regexp filter {:?}", tfs.get(0).unwrap());
        }
        must_add(tfs, "foo", ".*", true, true);
        expect_tag_filter(&tfs, 0, ".+", true, true);
        must_add(tfs, "foo", "foo||bar", false, true);
        expect_tag_filter(&tfs, 1, "foo||bar", false, true);

        // Verify that other filters are added normally.
        tfs.reset();
        must_add(tfs, "", "foobar", false, false);
        if tfs.len() != 1 {
            panic!("missing added filter")
        }
        must_add(tfs, "bar", "foobar", true, false);
        if tfs.len() != 2 {
            panic!("missing added filter")
        }
        must_add(tfs, "", "foo.+bar", true, true);
        if tfs.len() != 3 {
            panic!("missing added filter")
        }
        must_add(tfs, "bar", "foo.+bar", false, true);
        if tfs.len() != 4 {
            panic!("missing added filter")
        }
        must_add(tfs, "bar", "foo.*", false, true);
        if tfs.len() != 5 {
            panic!("missing added filter")
        }
    }
}