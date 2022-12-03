use crate::{MetricName, QueryResult, RuntimeResult, Timeseries};

pub fn test_results_equal(result: &Vec<QueryResult>, result_expected: &Vec<QueryResult>) {
    assert_eq!(
        result.len(),
        result_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        result.len(),
        result_expected.len()
    );

    for (i, actual) in result.iter().enumerate() {
        let r_expected = &result_expected.get(i).unwrap();
        test_metric_names_equal(&actual.metric_name, &r_expected.metric_name, i);
        test_rows_equal(
            &actual.values,
            &actual.timestamps,
            &r_expected.values,
            &r_expected.timestamps,
        );
    }
}

pub fn test_rows_equal(
    values: &[f64],
    timestamps: &[i64],
    values_expected: &[f64],
    timestamps_expected: &[i64],
) {
    assert_eq!(
        values.len(),
        values_expected.len(),
        "unexpected values.len(); got {}; want {}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
        values.len(),
        values_expected.len(),
        values,
        values_expected
    );
    assert_eq!(timestamps.len(), timestamps_expected.len(),
               "unexpected timestamps.len(); got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
               timestamps.len(), timestamps_expected.len(), timestamps, timestamps_expected);

    assert_eq!(
        values.len(),
        timestamps.len(),
        "values.len() doesn't match timestamps.len(); got {} vs {}",
        values.len(),
        timestamps.len()
    );

    let mut i = 0;
    while i < values.len() {
        let ts = timestamps[i];
        let ts_expected = timestamps_expected[i];
        assert_eq!(ts, ts_expected,
                   "unexpected timestamp at timestamps[{}]; got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
                   i, ts, ts_expected, timestamps, timestamps_expected);

        let v = values[i];
        let v_expected = values_expected[i];
        if v.is_nan() {
            assert!(v_expected.is_nan(),
                    "unexpected nan value at values[{}]; want %{}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, v_expected, values, values_expected);
            continue;
        }
        if v_expected.is_nan() {
            assert!(v.is_nan(), "unexpected value at values[{}]; got {}; want nan\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, v, values, values_expected);
            continue;
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

    for (i, tag) in mn.tags.iter().enumerate() {
        let tag_expected = &expected.tags[i];
        assert_eq!(
            tag.key, tag_expected.key,
            "unexpected tag key at #{},{}; got {}; want {}; got={}, expected={}",
            pos, i, tag.key, tag_expected.key, mn, expected
        );

        assert_eq!(
            tag.value, tag_expected.value,
            "unexpected tag value at #{},{}; got {}; want {}; got={}, expected={}",
            pos, i, tag.value, tag_expected.value, mn, expected
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

pub const EPSILON: f64 = 1e-14;

pub fn compare_values(vs1: &[f64], vs2: &[f64]) -> RuntimeResult<()> {
    assert_eq!(
        vs1.len(),
        vs2.len(),
        "unexpected number of values; got {}; want {}",
        vs1.len(),
        vs2.len()
    );
    for (i, v1) in vs1.iter().enumerate() {
        let v2 = &vs2[i];
        assert!(
            compare_floats(*v2, *v1),
            "unexpected value; got {}; want {}",
            v1,
            v2
        );
    }
    Ok(())
}

pub fn compare_floats(expected: f64, actual: f64) -> bool {
    if actual == expected {
        return true;
    }
    return match (expected.is_finite(), actual.is_finite()) {
        (true, true) => {
            let eps = (actual - expected).abs();
            eps <= EPSILON
        }
        (false, false) => {
            if expected.is_nan() {
                return actual.is_nan();
            }
            if expected == f64::INFINITY {
                return actual == f64::INFINITY;
            }
            return actual == f64::NEG_INFINITY;
        }
        _ => false,
    };
}

/// returns true if the two sample lines only differ by a
/// small relative error in their sample value.
pub(crate) fn almost_equal(a: f64, b: f64) -> bool {
    // NaN has no equality but for testing we still want to know whether both values
    // are NaN.
    if a.is_nan() && b.is_nan() {
        return true;
    }

    // Cf. http://floating-point-gui.de/errors/comparison/
    if a == b {
        return true;
    }

    let diff = (a - b).abs();
    let min_normal = f64::from_bits(0x0010000000000000); // The smallest positive normal value of type float64.
    if a == 0_f64 || b == 0_f64 || diff < min_normal {
        return diff < EPSILON * min_normal;
    }

    return diff / (a.abs() + b.abs()) < EPSILON;
}
