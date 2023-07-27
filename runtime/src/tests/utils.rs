use itertools::izip;

use crate::{MetricName, QueryResult, RuntimeResult, Timeseries};

pub fn test_results_equal(result: &Vec<QueryResult>, result_expected: &Vec<QueryResult>) {
    assert_eq!(
        result.len(),
        result_expected.len(),
        "unexpected timeseries count; got {}; want {}",
        result.len(),
        result_expected.len()
    );

    let mut i = 0;
    for (actual, expected) in result.iter().zip(result_expected) {
        test_metric_names_equal(&actual.metric_name, &expected.metric_name, i);
        test_rows_equal(
            &actual.values,
            &actual.timestamps,
            &expected.values,
            &expected.timestamps,
        );
        i = i + 1;
    }
}

pub fn test_rows_equal(
    values: &[f64],
    timestamps: &[i64],
    values_expected: &[f64],
    timestamps_expected: &[i64],
) {
    compare_values(values, values_expected).unwrap();
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

    for (i, val, val_expected, ts, ts_expected) in izip!(
        0..1000,
        values.iter(),
        values_expected.iter(),
        timestamps.iter(),
        timestamps_expected.iter()
    ) {
        assert_eq!(
            ts, ts_expected,
            "unexpected timestamp at timestamps[{}]; got {}; want {}\ntimestamps=\n{:?}\ntimestamps_expected=\n{:?}",
            i, ts, ts_expected, timestamps, timestamps_expected
        );

        if val.is_nan() {
            assert!(val_expected.is_nan(),
                    "unexpected nan value at values[{}]; want %{}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, val_expected, values, values_expected);
            continue;
        }
        if val_expected.is_nan() {
            assert!(val.is_nan(), "unexpected value at values[{}]; got {}; want nan\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                    i, val, values, values_expected);
            continue;
        }

        // Compare values with the reduced precision because of different precision errors
        // on different OS/architectures. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1738
        // and https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1653
        if (val - val_expected).abs() / (val_expected).abs() > 1e-13 {
            panic!("unexpected value at values[{}]; got {}; want {}\nvalues=\n{:?}\nvalues_expected=\n{:?}",
                   i, val, val_expected, values, values_expected)
        }
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

pub const EPSILON: f64 = 1e-14;

pub fn compare_values(actual: &[f64], expected: &[f64]) -> RuntimeResult<()> {
    assert_eq!(
        actual.len(),
        expected.len(),
        "unexpected number of values; got {}; want {}",
        actual.len(),
        expected.len()
    );
    for (i, (got, wanted)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            compare_floats(*wanted, *got),
            "unexpected value comparing slice; expected {v2}, got {v1} at index {i}",
            v1 = *got,
            v2 = *wanted
        );
    }
    Ok(())
}

pub fn compare_floats(expected: f64, actual: f64) -> bool {
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
