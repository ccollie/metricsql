#[cfg(test)]
mod tests {
    use crate::common::encoding::write_string;
    use crate::MetricName;

    #[test]
    fn test_metric_name_string() {
        fn f(mn: &MetricName, result_expected: &str) {
            let result = mn.to_string();
            assert_eq!(
                result, result_expected,
                "unexpected result\ngot\n{}\nwant\n{}",
                result, result_expected
            );
        }

        f(&MetricName::new("foobar"), "foobar{}");
        let mut mn = MetricName::new("abc");
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "123");
        f(&mn, r#"abc{baz="123",foo="bar"}"#)
    }

    #[test]
    fn test_metric_name_sort_tags() {
        check_metric_name_sort_tags(&[], &[]);
        check_metric_name_sort_tags(&["foo"], &["foo"]);
        check_metric_name_sort_tags(&["job"], &["job"]);
        check_metric_name_sort_tags(&["server"], &["server"]);
        check_metric_name_sort_tags(
            &["host", "foo", "bar", "service"],
            &["bar", "foo", "host", "service"],
        );
        check_metric_name_sort_tags(
            &["model", "foo", "job", "host", "server", "instance"],
            &["foo", "host", "instance", "job", "model", "server"],
        )
    }

    fn check_metric_name_sort_tags(tags: &[&str], expected: &[&str]) {
        let expected_tags = expected
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        let mut mn = MetricName::default();
        for t in tags.iter() {
            mn.add_tag(t, "");
        }
        mn.sort_tags();

        let result_tags = mn
            .tags
            .iter()
            .map(|x| x.key.clone())
            .collect::<Vec<String>>();

        assert_eq!(
            result_tags, expected_tags,
            "unexpected result_tags\ngot\n{:?}\nwant\n{:?}",
            result_tags, expected_tags
        );
    }

    #[test]
    fn test_metric_name_marshal_duplicate_keys() {
        let mut mn = MetricName::default();
        mn.metric_group = "xxx".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("duplicate", "tag1");
        mn.add_tag("duplicate", "tag2");
        mn.add_tag("tt", "xx");
        mn.add_tag("foo", "abc");
        mn.add_tag("duplicate", "tag3");

        let mut mn_expected = MetricName::default();
        mn_expected.metric_group = "xxx".to_string();
        mn_expected.add_tag("duplicate", "tag3");
        mn_expected.add_tag("foo", "abc");
        mn_expected.add_tag("tt", "xx");

        mn.sort_tags();
        let mut data: Vec<u8> = vec![];
        mn.marshal(&mut data);

        let (_, mn1) = MetricName::unmarshal(&mut data).expect("unmarshal");
        assert_eq!(
            &mn_expected, &mn1,
            "unexpected mn unmarshalled;\ngot\n{}\nwant\n{}",
            &mn1, &mn_expected
        );
    }

    #[test]
    fn test_metric_name_marshal_unmarshal() {
        for i in 0..10 {
            for tags_count in 0..10 {
                let mut mn = MetricName::default();
                for j in 0..tags_count {
                    mn.add_tag(
                        format!("key_{}_{}_\x00\x01\x02", i, j).as_str(),
                        format!("\x02\x00\x01value_{}_{}", i, j).as_str(),
                    );
                }

                let mut data: Vec<u8> = vec![];
                mn.marshal(&mut data);

                let (_, mn1) = MetricName::unmarshal(&mut &data).expect("unmarshal");
                assert_eq!(
                    mn, mn1,
                    "unexpected mn unmarshalled;\ngot\n{:?}\nwant\n{:?}",
                    &mn1, &mn
                );

                // Try unmarshalling MetricName without tag value.
                let mut broken_data = b"foobar".to_vec();
                match MetricName::unmarshal(&mut broken_data) {
                    Ok(_) => {
                        panic!("expecting non-zero error when unmarshalling MetricName without tag value")
                    }
                    _ => {}
                }

                let len = broken_data.len();
                // Try unmarshalling MetricName with invalid tag key.
                broken_data[len - 1] = 123;
                match MetricName::unmarshal(&mut broken_data) {
                    Ok(_) => {
                        panic!("expecting non-zero error when unmarshalling MetricName with invalid tag key")
                    }
                    _ => {}
                }

                // Try unmarshalling MetricName with invalid tag value.
                let mut broken_data = b"foobar".to_vec();
                write_string(&mut broken_data, "aaa");

                let len = broken_data.len();
                broken_data[len - 1] = 123;

                match MetricName::unmarshal(&mut broken_data) {
                    Ok(_) => {
                        panic!("expecting non-zero error when unmarshalling MetricName with invalid tag value")
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_metric_name_remove_tags_on() {
        let mut empty_mn = MetricName::new("name");
        empty_mn.add_tag("key", "value");
        empty_mn.remove_tags_on(&vec![]);
        if empty_mn.metric_group.len() != 0 || empty_mn.tags.len() != 0 {
            panic!("expecting empty metric name got {}", &empty_mn)
        }

        let mut as_is_mn = MetricName::new("name");
        as_is_mn.add_tag("key", "value");
        as_is_mn.remove_tags_on(&vec!["__name__".to_string(), "key".to_string()]);

        let mut exp_as_is_mn = MetricName::new("name");
        exp_as_is_mn.add_tag("key", "value");
        assert_eq!(
            exp_as_is_mn, as_is_mn,
            "expecting {} got {}",
            &exp_as_is_mn, &as_is_mn
        );

        let mut mn = MetricName::new("name");

        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        mn.remove_tags_on(&vec!["baz".to_string()]);

        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert_eq!(exp_mn, mn, "expecting {} got {}", &exp_mn, &mn);
    }

    #[test]
    fn test_metric_name_remove_tag() {
        let mut mn = MetricName::default();
        mn.metric_group = "name".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        mn.remove_tag("__name__");
        assert_eq!(
            mn.metric_group.len(),
            0,
            "expecting empty metric group got {}",
            &mn
        );
        mn.remove_tag("foo");

        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert!(
            names_equal(&mut exp_mn, &mut mn),
            "expecting {} got {}",
            &exp_mn,
            &mn
        )
    }

    #[test]
    fn test_metric_name_remove_tags_ignoring() {
        let mut mn = MetricName::default();
        mn.metric_group = "name".to_string();
        mn.add_tag("foo", "bar");
        mn.add_tag("baz", "qux");
        mn.remove_tags_ignoring(&vec!["__name__".to_string(), "foo".to_string()]);
        let mut exp_mn = MetricName::default();
        exp_mn.add_tag("baz", "qux");
        assert!(
            names_equal(&mut mn, &mut exp_mn),
            "expecting {} got {}",
            &exp_mn,
            &mn
        )
    }

    fn names_equal(a: &mut MetricName, b: &mut MetricName) -> bool {
        if a.tags.len() != b.tags.len() {
            return false;
        }

        a.sort_tags();
        b.sort_tags();

        for (x, y) in a.tags.iter().zip(&b.tags) {
            if x.key != y.key || x.value != y.value {
                return false;
            }
        }

        true
    }
}
