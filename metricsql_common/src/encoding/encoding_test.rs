#[cfg(test)]
mod tests {
    use crate::encoding::is_const;
    use crate::{
        ensure_non_decreasing_sequence, is_delta_const, is_gauge,
    };

    #[test]
    fn test_is_const() {
        let f = |a: &[i64], ok_expected: bool| {
            let ok = is_const(a);
            assert_eq!(
                ok, ok_expected,
                "unexpected is_const for a={:?}; got {}; want {}",
                a, ok, ok_expected
            );
        };
        f(&[], false);
        f(&[1], true);
        f(&[1, 2], false);
        f(&[1, 1], true);
        f(&[1, 1, 1], true);
        f(&[1, 1, 2], false);
    }

    #[test]
    fn test_is_delta_const() {
        let f = |a: &[i64], ok_expected: bool| {
            let ok = is_delta_const(a);
            assert_eq!(
                ok, ok_expected,
                "unexpected isDeltaConst for a={:?}; got {}; want {}",
                a, ok, ok_expected
            );
        };

        f(&[], false);
        f(&[1], false);
        f(&[1, 2], true);
        f(&[1, 2, 3], true);
        f(&[3, 2, 1], true);
        f(&[3, 2, 1, 0, -1, -2], true);
        f(&[3, 2, 1, 0, -1, -2, 2], false);
        f(&[1, 1], true);
        f(&[1, 2, 1], false);
        f(&[1, 2, 4], false);
    }

    #[test]
    fn test_is_gauge() {
        let f = |values: &[i64], ok_expected: bool| {
            let ok = is_gauge(values);
            assert_eq!(
                ok, ok_expected,
                "unexpected result for is_gauge({:?}); got {}; expecting {}",
                values, ok, ok_expected
            );
        };
        f(&[], false);
        f(&[0], false);
        f(&[1, 2], false);
        f(&[0, 1, 2, 3, 4, 5], false);
        f(&[0, -1, -2, -3, -4], true);
        f(&[0, 0, 0, 0, 0, 0, 0], false);
        f(&[1, 1, 1, 1, 1], false);
        f(&[1, 1, 2, 2, 2, 2], false);
        f(&[1, 17, 2, 3], false); // a single counter reset
        f(&[1, 5, 2, 3], true);
        f(&[1, 5, 2, 3, 2], true);
        f(&[-1, -5, -2, -3], true);
        f(&[-1, -5, -2, -3, -2], true);
        f(&[5, 6, 4, 3, 2], true);
        f(&[4, 5, 6, 5, 4, 3, 2], true);
        f(&[1064, 1132, 1083, 1062, 856, 747], true);
    }

    #[test]
    fn test_ensure_non_decreasing_sequence() {
        _ensure_non_decreasing_sequence(&[], -1234, -34, &[]);
        _ensure_non_decreasing_sequence(&[123], -1234, -1234, &[-1234]);
        _ensure_non_decreasing_sequence(&[123], -1234, 345, &[345]);
        _ensure_non_decreasing_sequence(&[-23, -14], -23, -14, &[-23, -14]);
        _ensure_non_decreasing_sequence(&[-23, -14], -25, 0, &[-25, 0]);
        _ensure_non_decreasing_sequence(&[0, -1, 10, 5, 6, 7], 2, 8, &[2, 2, 8, 8, 8, 8]);
        _ensure_non_decreasing_sequence(&[0, -1, 10, 5, 6, 7], -2, 8, &[-2, -1, 8, 8, 8, 8]);
        _ensure_non_decreasing_sequence(&[0, -1, 10, 5, 6, 7], -2, 12, &[-2, -1, 10, 10, 10, 12]);
        _ensure_non_decreasing_sequence(&[1, 2, 1, 3, 4, 5], 1, 5, &[1, 2, 2, 3, 4, 5]);
    }

    fn _ensure_non_decreasing_sequence(values: &[i64], min: i64, max: i64, expected: &[i64]) {
        let mut values = Vec::from(values);
        ensure_non_decreasing_sequence(&mut values, min, max);
        assert_eq!(
            values, expected,
            "unexpected a; got\n{:?}; expecting\n{:?}",
            values, expected
        );
    }

}
