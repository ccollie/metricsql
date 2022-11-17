#[cfg(test)]
mod tests {
    use metricsql::ast::MetricExpr;
    use crate::{Rows, Timeseries};
    use crate::eval::binary_op::get_common_label_filters;

    #[test]
    fn test_get_common_label_filters() {
        let f = |metrics: &str, lfs_expected: &str| {
            let mut tss: Vec<Timeseries> = vec![];

            let mut rows = Rows::try_from(metrics)
                .expect("error initializing rows from string");

            match rows.unmarshal(metrics) {
                Err(err) => {
                    panic!("unexpected error when parsing {}: {:?}", metrics, err);
                }
                Ok(_) => {}
            }
            for row in rows.iter() {
                let mut ts = Timeseries::default();
                for tag in row.tags.iter() {
                    ts.metric_name.set_tag(&tag.key, &tag.value);
                }
                tss.push(ts)
            }

            let lfs = get_common_label_filters(&tss);
            let me = MetricExpr::with_filters(lfs);

            let lfs_marshaled = me.to_string();
            assert_eq!(lfs_marshaled, lfs_expected,
                       "unexpected common label filters;\ngot\n{}\nwant\n{}", lfs_marshaled, lfs_expected)
        };

        f("", "{}");
        f("m 1", "{}");
        f(r#"m { a="b" } 1"#, r#"{a = "b"}"#);
        f(r#"m { c="d", a="b" } 1"#, r#"{a = "b", c = "d"}"#);
        f(r#"m1 { a="foo" } 1
          m2 { a="bar" } 1"#, r#"{a = ~"bar|foo"}"#);
        f(r#"m1 { a="foo" } 1
          m2 { b="bar" } 1"#, "{}");
        f(r#"m1 { a="foo", b="bar" } 1
          m2 { b="bar", c="x" } 1"#, r#"{b = "bar"}"#);
    }
}