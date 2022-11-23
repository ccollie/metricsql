#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::{Timeseries};
    use crate::functions::transform::transform_fns::{fix_broken_buckets, LeTimeseries};
    use crate::functions::transform::vmrange_buckets_to_le;
    use crate::prometheus::*;

    const NAN: f64 = f64::NAN;

    fn check_broken_buckets(values: &[f64], expected: &[f64])
    {
        let values = Vec::from(values);
        let mut xss: Vec<LeTimeseries> = Vec::with_capacity(values.len());
        for v in values.iter() {
            let ts = LeTimeseries {
                le: 0.0,
                ts: Timeseries::new(vec![1000], vec![*v])
            };
            xss.push(ts);
        }

        fix_broken_buckets(0, &mut xss);

        let result = xss.iter()
            .map(|xs| xs.ts.values[0])
            .collect::<Vec<f64>>();

        assert_eq!(expected, &result,
            "unexpected result for values={:?}\ngot\n{:?}\nwant\n{:?}", values, result, expected)
    }

    #[test]
    fn test_fix_broken_buckets() {
        check_broken_buckets(&[], &[]);
        check_broken_buckets( &[1.0], &[1.0]);
        check_broken_buckets(&[1.0, 2.0], &[1.0, 2.0]);
        check_broken_buckets(&[2.0, 1.0], &[1.0, 1.0]);
        check_broken_buckets(&[1.0, 2.0, 3.0, NAN, NAN], &[1.0, 2.0, 3.0, 3.0, 3.0]);
        check_broken_buckets(&[5.0, 1.0, 2.0, 3.0, NAN], &[1.0, 1.0, 2.0, 3.0, 3.0]);
        check_broken_buckets(&[1.0, 5.0, 2.0, NAN, 6.0, 3.0], &[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
        check_broken_buckets(&[5.0, 10.0, 4.0, 3.0], &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_vmrange_buckets_to_le() {
        let f = |buckets: &str, buckets_expected: &str| {
            let tss = prom_metrics_to_timeseries(buckets);
            let result = vmrange_buckets_to_le(tss);
            let result_buckets = timeseries_to_prom_metrics(&result);
            assert_eq!(result_buckets, buckets_expected,
                       "unexpected vmrange_buckets_to_le(); got\n{:?}\nwant\n{}", result_buckets, buckets_expected);
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

    // #[test]
    // fn limit_offset_too_big_offset() {
    //     let q = r#"limit_offset(1, 10, sort_by_label((
    //                         label_set(time()*1, "foo", "y"),
    //                         label_set(time()*2, "foo", "a"),
    //                         label_set(time()*3, "foo", "x"),
    //                     ), "foo"))"#;
    //     let result_expected = QueryResult::default();
    //     f(q, result_expected);
    // }

    fn prom_metrics_to_timeseries(s: &str) -> Vec<Timeseries> {
        let mut rows = Rows::default();
        rows.unmarshal(s).expect(&*format!("cannot parse {}", s));
        let mut tss: Vec<Timeseries> = vec![];

        for row in rows.iter() {
            let mut ts: Timeseries = Timeseries::default();
            ts.metric_name.metric_group = row.metric.clone();
            for tag in row.tags.iter() {
                ts.metric_name.set_tag(&tag.key, &tag.value)
            }
            let timestamps = vec![row.timestamp/1000];
            ts.timestamps = Arc::new(timestamps);
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
                a.push(line);
            }
        }
        return a.join("\n")
    }
}