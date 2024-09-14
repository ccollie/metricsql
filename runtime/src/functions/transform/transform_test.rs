#[cfg(test)]
mod tests {
    use crate::functions::transform::histogram::{fix_broken_buckets, LeTimeseries};
    use crate::functions::transform::vmrange_buckets_to_le;
    use crate::prometheus_parse::*;
    use crate::types::Timeseries;

    const NAN: f64 = f64::NAN;

    fn check_broken_buckets(values: &[f64], expected: &[f64]) {
        let values = Vec::from(values);
        let mut xss: Vec<LeTimeseries> = Vec::with_capacity(values.len());
        for v in values.iter() {
            let ts = LeTimeseries {
                le: 0.0,
                ts: Timeseries::new(vec![1000], vec![*v]),
            };
            xss.push(ts);
        }

        fix_broken_buckets(0, &mut xss);

        let result = xss.iter().map(|xs| xs.ts.values[0]).collect::<Vec<f64>>();

        assert_eq!(
            expected, &result,
            "unexpected result for values={:?}\ngot\n{:?}\nwant\n{:?}",
            values, result, expected
        )
    }

    #[test]
    fn test_fix_broken_buckets() {
        check_broken_buckets(&[], &[]);
        check_broken_buckets(&[1.0], &[1.0]);
        check_broken_buckets(&[1.0, 2.0], &[1.0, 2.0]);
        check_broken_buckets(&[2.0, 1.0], &[2.0, 2.0]);
        check_broken_buckets(&[1.0, 2.0, 3.0, NAN, NAN], &[1.0, 2.0, 3.0, 3.0, 3.0]);
        check_broken_buckets(&[5.0, 1.0, 2.0, 3.0, NAN], &[5.0, 5.0, 5.0, 5.0, 5.0]);
        check_broken_buckets(
            &[1.0, 5.0, 2.0, NAN, 6.0, 3.0],
            &[1.0, 5.0, 5.0, 5.0, 6.0, 6.0],
        );
        check_broken_buckets(&[5.0, 10.0, 4.0, 3.0], &[5.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_fix_broken_buckets_multiple_values() {
        fn f(values: Vec<Vec<f64>>, expected_result: Vec<Vec<f64>>) {
            let mut xss: Vec<LeTimeseries> = Vec::with_capacity(values.len());
            for (i, v) in values.iter().enumerate() {
                let mut timestamps = Vec::with_capacity(v.len());
                for _ in 0..v.len() {
                    timestamps.push(1000 + i as i64);
                }
                let ts = Timeseries::new(timestamps, v.clone());
                xss.push(LeTimeseries { le: 0.0, ts });
            }
            for i in 0..values.len() {
                fix_broken_buckets(i, &mut xss)
            }
            let mut result: Vec<Vec<f64>> = Vec::with_capacity(values.len());
            for xs in xss.into_iter() {
                result.push(xs.ts.values)
            }
            assert_eq!(
                result, expected_result,
                "unexpected result for values={:?}\ngot\n{:?}\nwant\n{:?}",
                values, result, expected_result
            );
        }

        let values = vec![vec![10.0, 1.0], vec![11.0, 2.0], vec![13.0, 3.0]];
        let expected_result = vec![vec![10.0, 1.0], vec![11.0, 2.0], vec![13.0, 3.0]];
        f(values, expected_result);
    }

    fn check_vmrange_buckets_to_le(buckets: &str, buckets_expected: &str) {
        let tss = prom_metrics_to_timeseries(buckets);
        let result = vmrange_buckets_to_le(tss);
        let result_buckets = timeseries_to_prom_metrics(&result);
        assert_eq!(
            result_buckets, buckets_expected,
            "unexpected vmrange_buckets_to_le(); got\n{:?}\nwant\n{}",
            result_buckets, buckets_expected
        );
    }

    #[test]
    fn test_vmrange_buckets_to_le_single_non_empty_bucket() {
        // A single non-empty vmrange bucket
        check_vmrange_buckets_to_le(
            r#"foo{vmrange="4.084e+02...4.642e+02"} 2 123"#,
            r#"foo{le="4.084e+02"} 0 123
        foo{le="4.642e+02"} 2 123
        foo{le="+Inf"} 2 123"#,
        );

        check_vmrange_buckets_to_le(
            r#"foo{vmrange="0...+Inf"} 5 123"#,
            r#"foo{le="+Inf"} 5 123"#,
        );

        check_vmrange_buckets_to_le(
            r#"foo{vmrange="-Inf...0"} 4 123"#,
            r#"foo{le="-Inf"} 0 123
        foo{le="0"} 4 123
        foo{le="+Inf"} 4 123"#,
        );

        check_vmrange_buckets_to_le(
            r#"foo{vmrange="-Inf...+Inf"} 1.23 456"#,
            r#"foo{le="-Inf"} 0 456
        foo{le="+Inf"} 1.23 456"#,
        );

        check_vmrange_buckets_to_le(
            r#"foo{vmrange="0...0"} 5.3 0"#,
            r#"foo{le="0"} 5.3 0
        foo{le="+Inf"} 5.3 0"#,
        );
    }

    #[test]
    fn test_vmrange_buckets_to_le() {
        let f = |buckets: &str, buckets_expected: &str| {
            let tss = prom_metrics_to_timeseries(buckets);
            let result = vmrange_buckets_to_le(tss);
            let result_buckets = timeseries_to_prom_metrics(&result);
            assert_eq!(
                result_buckets, buckets_expected,
                "unexpected vmrange_buckets_to_le(); got\n{:?}\nwant\n{}",
                result_buckets, buckets_expected
            );
        };

        // Adjacent empty vmrange bucket
        f(
            r#"foo{vmrange="7.743e+05...8.799e+05"} 5 123
            foo{vmrange="6.813e+05...7.743e+05"} 0 123"#,
            r#"foo{le="7.743e+05"} 0 123
            foo{le="8.799e+05"} 5 123
            foo{le="+Inf"} 5 123"#,
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
        f(r#"foo{vmrange="0...1"} 0 123"#, r#""#);

        f(r#"foo{vmrange="0...+Inf"} 0 123"#, "");

        f(r#"foo{vmrange="-Inf...0"} 0 123"#, "");
        f(r#"foo{vmrange="0...0"} 0 0"#, "");
        f(r#"foo{vmrange="-Inf...+Inf"} 0 456"#, "");

        // Multiple empty vmrange buckets
        f(
            r#"foo{vmrange="2...3"} 0 123
        foo{vmrange="1...2"} 0 123"#,
            "",
        );

        // The bucket with negative value
        f(r#"foo{vmrange="4.084e+02...4.642e+02"} -5 1"#, "");

        // Missing vmrange in the original metric
        f(r#"foo 3 6"#, "");

        // Missing le label in the original metric
        f(r#"foo{le="456"} 3 6"#, r#"foo{le="456"} 3 6"#);

        // Invalid vmrange label value
        f(r#"foo{vmrange="foo...bar"} 1 1"#, "");

        f(r#"foo{vmrange="4.084e+02"} 1 1"#, "");

        f(r#"foo{vmrange="4.084e+02...foo"} 1 1"#, "")
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
        let _rows = s
            .split("\r\n")
            .filter(|x| x.len() > 0)
            .map(LineInfo::parse)
            .collect::<Vec<LineInfo>>();

        let tss: Vec<Timeseries> = vec![];
        // for row in rows.iter() {
        //     match row {
        //         LineInfo::Sample(sample) => {
        //             let mut ts: Timeseries = Timeseries::default();
        //             ts.metric_name.metric_group = row.metric_name.clone();
        //             for (key, value) in row.labels.iter() {
        //                 ts.metric_name.set_tag(&key, &value)
        //             }
        //             let timestamps = vec![row.timestamp / 1000];
        //             ts.timestamps = Arc::new(timestamps);
        //             ts.values.push(row.value);
        //             tss.push(ts);
        //         },
        //         _=> continue
        //     }
        // }
        tss
    }

    fn timeseries_to_prom_metrics(tss: &[Timeseries]) -> String {
        let mut a: Vec<String> = vec![];
        for ts in tss.iter() {
            let metric_name = ts.metric_name.to_string();
            for i in 0..ts.timestamps.len() {
                let line = format!("{} {} {}", metric_name, ts.values[i], ts.timestamps[i]);
                a.push(line);
            }
        }
        a.join("\n")
    }
}
