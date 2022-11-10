#[cfg(test)]
mod tests {
    pub fn test_results_equal(result: &Vec<QueryResult>, result_expected: &Vec<QueryResult>) {

        assert_eq!(result.len(), result_expected.len(),
                   "unexpected timeseries count; got {}; want {}", result.len(), result_expected.len());

        for (i, actual) in result.iter().enumerate() {
            let r_expected = &result_expected.get(i);
            test_metric_names_equal(&actual.metric_name, &r_expected.metric_name, i);
            test_rows_equal(&actual.values, &actual.timestamps, r_expected.values, r_expected.timestamps);
        }
    }

    pub fn test_rows_equal(values: &[f64], timestamps: &[i64], values_expected: &[f64], timestamps_expected: &[i64]) {
        assert_eq!(values.len(), values_expected.len(),
                   "unexpected values.len(); got {}; want {}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                   values.len(), values_expected.len(), values, values_expected);
        assert_eq!(timestamps.len(), timestamps_expected.len(),
                   "unexpected timestamps.len(); got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
                   timestamps.len(), timestamps_expected.len(), timestamps, timestamps_expected);

        assert_eq!(values.len(), timestamps.len(),
                   "values.len() doesn't match timestamps.len(); got %d vs %d", values.len(), timestamps.len());

        let mut i = 0;
        while i < values.len() {
            let ts = timestamps[i];
            let ts_expected = timestamps_expected[i];
            assert_eq!(ts, ts_expected,
                       "unexpected timestamp at timestamps[{}]; got {}; want {}\ntimestamps=\n{}\ntimestamps_expected=\n{}",
                       i, ts, ts_expected, timestamps, timestampsExpected: timestamps_expected);

            let v = values[i];
            let v_expected = values_expected[i];
            if v.is_nan() {
                assert!(v_expected.is_nan(),
                        "unexpected nan value at values[{}]; want %{}\nvalues=\n{}\nvalues_expected=\n{}",
                        i, v_expected, values, values_expected);
                continue;
            }
            if v_expected.is_nan() {
                assert!(v.is_nan(), "unexpected value at values[{}]; got {}; want nan\nvalues=\n{}\nvalues_expected=\n{}",
                        i, v, values, values_expected);
                continue
            }
            // Compare values with the reduced precision because of different precision errors
            // on different OS/architectures. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1738
            // and https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1653
            if (v - v_expected).abs() / (v_expected).abs() > 1e-13 {
                panic!("unexpected value at values[{}]; got {}; want {}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                       i, v, v_expected, values, values_expected)
            }
            i += 1;
        }
    }

    pub fn test_metric_names_equal(mn: &MetricName, expected: &MetricName, pos: usize) {
        assert_eq!(mn.metric_group, expected.metric_group,
                   "unexpected MetricGroup at #{}; got {}; want {}; metricGot={}, metricExpected={}",
                   pos, mn.metric_group, expected.metric_group, mn, expected);

        assert_eq!(mn.tag_count(), expected.tag_count(),
                   "unexpected tags count at #{}; got {}; want {}; metricGot={}, metricExpected={}",
                   pos, mn.tag_count(), expected.tag_count(), mn, expected);

        let tags_a = mn.get_tags();
        let tags_b = expected.get_tags();

        for (i, tag) in tags_a.iter().enumerate() {
            let tag_expected = &tags_b[i];
            assert_eq!(tag.key, tag_expected.key,
                       "unexpected tag key at #{},{}; got {}; want {}; metricGot={}, metricExpected={}",
                       pos, i, tag.key, tag_expected.key, mn, expected);

            assert_eq!(tag.value, tag_expected.value,
                       "unexpected tag value at #{},{}; got {}; want {}; metricGot={}, metricExpected={}",
                       pos, i, tag.value, tag_expected.value, mn, expected)
        }
    }


    pub fn compare_values<T>(vs1: T, vs2: T) -> RuntimeResultM<()> {
        assert_eq!(vs1.len(), vs2.len(),
            "unexpected number of values; got %d; want %d", vs1.len(), vs2.len());
        for (i, v1) in vs1.iter().enumerate() {
            let v2 = vs2[i];
            if v1.is_nan() {
                assert!(v2.is_nan(), "unexpected value; got {}; want {}", v1, v2);
            }
            continue
        }
        let eps = (v1 - v2).abs();
        assert!(eps <= 1e-14,  "unexpected value; got {}; want {}", v1, v2);
        Ok(())
    }
}