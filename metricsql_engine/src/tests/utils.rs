use crate::{MetricName, QueryResult, Timeseries};
use metricsql_testing::test_rows_equal;

pub fn test_results_equal(result: &[QueryResult], result_expected: &[QueryResult]) {
    assert_eq!(
        result.len(),
        result_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        result.len(),
        result_expected.len()
    );

    let mut i = 0;
    for (actual, expected) in result.iter().zip(result_expected) {
        test_metric_names_equal(&actual.metric, &expected.metric, i);
        test_rows_equal(
            &actual.values,
            &actual.timestamps,
            &expected.values,
            &expected.timestamps,
        );
        i = i + 1;
    }
}

pub fn test_metric_names_equal(mn: &MetricName, expected: &MetricName, pos: usize) {
    assert_eq!(
        mn.metric_group, expected.metric_group,
        "unexpected MetricGroup at #{}; got {}; want {}; metricGot={}, metricExpected={}",
        pos, mn.metric_group, expected.metric_group, mn, expected
    );

    assert_eq!(
        mn.tags.len(),
        expected.tags.len(),
        "unexpected tags count at #{}; got {}; want {}; metricGot={}, metricExpected={}",
        pos,
        mn.tags.len(),
        expected.tags.len(),
        mn,
        expected
    );

    for (i, (tag, tag_expected)) in mn.tags.iter().zip(expected.tags.iter()).enumerate() {
        assert_eq!(
            tag.key, tag_expected.key,
            "unexpected tag key at #{pos}, {i}; got {}; want {}; got={mn}, expected={expected}",
            tag.key, tag_expected.key
        );

        assert_eq!(
            tag.value, tag_expected.value,
            "unexpected tag value at #{pos},{i}; got {}; want {}; got={mn}, expected={expected}",
            tag.value, tag_expected.value,
        )
    }
}

pub fn test_timeseries_equal(tss: &[Timeseries], tss_expected: &[Timeseries]) {
    assert_eq!(
        tss.len(),
        tss_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        tss.len(),
        tss_expected.len()
    );

    for (i, ts) in tss.iter().enumerate() {
        let ts_expected = &tss_expected[i];
        test_metric_names_equal(&ts.metric_name, &ts_expected.metric_name, i);
        test_rows_equal(
            &ts.values,
            &ts.timestamps,
            &ts_expected.values,
            &ts_expected.timestamps,
        )
    }
}
