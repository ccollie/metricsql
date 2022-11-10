#[cfg(test)]
mod tests {
    use crate::{Tag, Timeseries};
    use crate::functions::transform::vmrange_buckets_to_le;
    use crate::prometheus::*;


    fn check_broken_buckets<T>(values: T, expected: T)
    where T: Into<Vec<f64>>
    {
        let values = values.into();
        let mut xss = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            let mut ts = Timeseries::new(vec![1000], vec![*v]);
            xss.push(ts);
        }
        fix_broken_buckets(0, xss);
        let result = Vec::with_capacity(values.len());
        for (i, xs) in xss.iter().enumerate() {
            result[i] = xs.ts.values[0];
        }
        let expected_result = expected.into();
        assert_eq!(expected_result, result,
            "unexpected result for values={}\ngot\n{}\nwant\n{}", values, result, expected_result)
    }
    #[test]
    fn test_fix_broken_buckets() {
        check_broken_buckets(&[], &[]);
        check_broken_buckets( &[1], &[1]);
        check_broken_buckets(&[1, 2], &[1, 2]);
        check_broken_buckets(&[2, 1], &[1, 1]);
        check_broken_buckets(&[1, 2, 3, nan, nan], &[1, 2, 3, 3, 3]);
        check_broken_buckets(&[5, 1, 2, 3, nan], &[1, 1, 2, 3, 3]);
        check_broken_buckets(&[1, 5, 2, nan, 6, 3], &[1, 2, 2, 3, 3, 3]);
        check_broken_buckets(&[5, 10, 4, 3], &[3, 3, 3, 3]);
    }

    #[test]
    fn test_vmrange_buckets_to_le() {
        let f = |buckets: &str, buckets_expected: &str| {
            let tss = prom_metrics_to_timeseries(buckets);
            let result = vmrange_buckets_to_le(tss);
            let result_buckets = timeseries_to_prom_metrics(&result);
            if !reflect.DeepEqual(result_buckets, buckets_expected) {
                t.Errorf("unexpected vmrangeBucketsToLE(); got\n{:?}\nwant\n{}", result_buckets, buckets_expected)
            }
        };

        // A single non-empty vmrange bucket
        f(
        r#"foo{vmrange="4.084e+02...4.642e+02"} 2 123"#,
        r#"foo{le="4.084e+02"} 0 123
        foo{le="4.642e+02"} 2 123
        foo{le="+Inf"} 2 123"#,
        );
        
        f(
        r#"foo{vmrange="0...+Inf"} 5 123"#,
        r#"foo{le="+Inf"} 5 123"#,
        );
        
        f(
        r#"foo{vmrange="-Inf...0"} 4 123"#,
        r#"foo{le="-Inf"} 0 123
        foo{le="0"} 4 123
        foo{le="+Inf"} 4 123"#,
        );
        
        f(
        r#"foo{vmrange="-Inf...+Inf"} 1.23 456"#,
        r#"foo{le="-Inf"} 0 456
        foo{le="+Inf"} 1.23 456"#,
        );
        
        f(
        r#"foo{vmrange="0...0"} 5.3 0"#,
        r#"foo{le="0"} 5.3 0
        foo{le="+Inf"} 5.3 0"#,
        );

        // Multiple non-empty vmrange buckets
        f(
        r#"foo{vmrange="4.084e+02...4.642e+02"} 2 123
        foo{vmrange="1.234e+02...4.084e+02"} 3 123
        "#,
        r#"foo{le="1.234e+02"} 0 123
        foo{le="4.084e+02"} 3 123
        foo{le="4.642e+02"} 5 123
        foo{le="+Inf"} 5 123"#,
        );

        // Multiple disjoint vmrange buckets
        f(
        r#"foo{vmrange="1...2"} 2 123
        foo{vmrange="4...6"} 3 123
        "#,
        r#"foo{le="1"} 0 123
        foo{le="2"} 2 123
        foo{le="4"} 2 123
        foo{le="6"} 5 123
        foo{le="+Inf"} 5 123"#,
        );

        // Multiple intersected vmrange buckets
        f(
        r#"foo{vmrange="1...5"} 2 123
        foo{vmrange="4...6"} 3 123"#,
        r#"foo{le="1"} 0 123
        foo{le="5"} 2 123
        foo{le="4"} 2 123
        foo{le="6"} 5 123
        foo{le="+Inf"} 5 123"#,
        );

        // Multiple vmrange buckets with the same end range
        f(
        r#"foo{vmrange="1...5"} 2 123
        foo{vmrange="0...5"} 3 123
        "#,
        r#"foo{le="1"} 0 123
        foo{le="5"} 2 123
        foo{le="0"} 2 123
        foo{le="+Inf"} 2 123"#,
        );

        // A single empty vmrange bucket
        f(
        r#"foo{vmrange="0...1"} 0 123"#,
        r#""#,
        );
        
        f(
        r#"foo{vmrange="0...+Inf"} 0 123"#,
        "",
        );
        
        f(r#"foo{vmrange="-Inf...0"} 0 123"#,"");
        f(r#"foo{vmrange="0...0"} 0 0"#, "");
        f(r#"foo{vmrange="-Inf...+Inf"} 0 456"#,"");

        // Multiple empty vmrange buckets
        f(
        r#"foo{vmrange="2...3"} 0 123
        foo{vmrange="1...2"} 0 123"#,
        "",
        );

        // The bucket with negative value
        f(
        r#"foo{vmrange="4.084e+02...4.642e+02"} -5 1"#,
        "",
        );

        // Missing vmrange in the original metric
        f(
        r#"foo 3 6"#,
        "",
        );

        // Missing le label in the original metric
        f(
        r#"foo{le="456"} 3 6"#,
        r#"foo{le="456"} 3 6"#,
        );
        
        // Invalid vmrange label value
        f(
        r#"foo{vmrange="foo...bar"} 1 1"#,
        "",
        );
        
        f(
        r#"foo{vmrange="4.084e+02"} 1 1"#,
        "",
        );
        
        f(
        r#"foo{vmrange="4.084e+02...foo"} 1 1"#,
        "",
        )
}

    fn prom_metrics_to_timeseries(s: &str) -> Vec<Timeseries> {
        let rows = prometheus::Rows::defauult();
        rows.unmarshal(s).expect(format!("cannot parse {}", s));
        let mut tss: Vec<Timeseries> = vec![];
        for row in rows.Rows.iter() {
            let mut tags: Vec<Tag> = vec![];
            for tag in row.Tags {
                tags.push( Tag{
                    key: tag.key.as_bytes(),
                    value: tag.value.as_bytes(),
                })
            }
            let mut ts: Timeseries = Timeseries::default();
            ts.metric_name.metric_group = row.metric;
            ts.metric_name.tags = tags;
            ts.timestamps.push(row.timestamp/1000);
            ts.values.push(row.value);
            tss.push(ts);
        }
        return tss
    }


fn timeseries_to_prom_metrics(tss: &[Timeseries]) -> String {
    let mut a: Vec<String> = vec![];
    for ts in tss.iter() {
        let metric_name = ts.metric_name.to_string();
        for i in 0 .. ts.timestamps.len() {
            let line = format!("{} {} {}", metric_name, ts.values[i], ts.timestamps[i]);
            a.push(linee);
        }
    }
    return a.join("\n")
}


}