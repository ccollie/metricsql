use itertools::izip;

pub fn test_rows_equal(
    values: &[f64],
    timestamps: &[i64],
    values_expected: &[f64],
    timestamps_expected: &[i64],
) {
    compare_values(values, values_expected);
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

pub const EPSILON: f64 = 1e-14;

pub fn compare_values(actual: &[f64], expected: &[f64]) {
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
            "unexpected value comparing slice; expected {v2}, got {v1} at index {i}\nactual={actual:?}\nexpected={expected:?}",
            v1 = *got,
            v2 = *wanted
        );
    }
}

pub fn compare_floats(expected: f64, actual: f64) -> bool {
    match (expected.is_finite(), actual.is_finite()) {
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
            actual == f64::NEG_INFINITY
        }
        _ => false,
    }
}
