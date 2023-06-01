#[cfg(test)]
mod tests {
    use crate::functions::mode_no_nans;

    const NAN: f64 = f64::NAN;

    #[test]
    fn test_mode_no_nans() {
        let f = |prev_value: f64, a: &[f64], expected_result: f64| {
            let mut values = Vec::from(a);
            let result = mode_no_nans(prev_value, &mut values);
            if result.is_nan() {
                assert!(
                    expected_result.is_nan(),
                    "unexpected result; got {}; want {}",
                    result,
                    expected_result
                );
                return;
            }
            if result != expected_result {
                panic!(
                    "unexpected result; got {}; want {}",
                    result, expected_result
                )
            }
        };

        f(NAN, &[], NAN);
        f(NAN, &[123.0], 123.0);
        f(NAN, &[1.0, 2.0, 3.0], 1.0);
        f(NAN, &[1.0, 2.0, 2.0], 2.0);
        f(NAN, &[1.0, 1.0, 2.0], 1.0);
        f(NAN, &[1.0, 1.0, 1.0], 1.0);
        f(NAN, &[1.0, 2.0, 2.0, 3.0], 2.0);
        f(NAN, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], 3.0);
        f(1.0, &[2.0, 3.0, 4.0, 5.0], 1.0);
        f(1.0, &[2.0, 2.0], 2.0);
        f(1.0, &[2.0, 3.0, 3.0], 3.0);
        f(1.0, &[2.0, 4.0, 3.0, 4.0, 3.0, 4.0], 4.0);
        f(1.0, &[2.0, 3.0, 3.0, 4.0, 4.0], 3.0);
        f(1.0, &[4.0, 3.0, 2.0, 3.0, 4.0], 3.0);
    }
}
