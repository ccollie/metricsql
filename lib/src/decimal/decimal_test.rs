#[cfg(test)]
mod tests {
    use crate::{append_decimal_to_float, append_float_to_decimal, calibrate_scale, from_float, is_stale_nan, max_up_exponent, positive_float_to_decimal, round_to_decimal_digits, STALE_NAN, STALE_NAN_BITS, to_float};
    use crate::decimal::{CONVERSION_PRECISION, V_INF_NEG, V_INF_POS, V_MAX, V_MIN};

    #[test]
    fn test_round_to_decimal_digits() {
        let f = |f: f64, digits: u8, result_expected: f64| {
            let result = round_to_decimal_digits(f, digits);
            if result.is_nan() {
                if is_stale_nan(result_expected) {
                    assert!(is_stale_nan(result),
                            "unexpected stale mark value; got {}; want {}", result.to_bits(), STALE_NAN_BITS);
                    return
                }
                assert!(result_expected.is_nan(), "unexpected result; got {}; want {}", result, result_expected);
                return
            }
            assert_eq!(result, result_expected, "unexpected result; got {}; want {}", result, result_expected);
        };

        f(12.34, 0, 12.0);
        f(12.57, 0, 13.0);
        f(-1.578, 2, -1.58);
        f(-1.578, 3, -1.578);
        f(1234.0, -2, 1200.0);
        f(1235.0, -1, 1240.0);
        f(1234.0, 0, 1234.0);
        f(1234.6, 0, 1235.0);
        f(123.4e-99, 99, 123e-99);
        f(NAN, 10, NAN);
        f(*STALE_NAN, 10, *STALE_NAN)
    }

    const NAN: f64 = f64::NAN;

    fn test_round_to_significant_figures() {
        let f = |f: f64, digits: u8, result_expected: f64| {
            let result = round_to_significant_figures(f, digits);
            if result.is_nan() {
                if is_stale_nan(result_expected) {
                    assert!(is_stale_nan(result), "unexpected stale mark value; got {}; want {}",
                            result.to_bits(), STALE_NAN_BITS);
                    return;
                }
                assert!(result_expected.is_nan(), "unexpected result; got {}; want {}", result, result_expected);
                return
            }
            assert_eq!(result, result_expected, "unexpected result; got {}; want {}", result, result_expected)
        };
        f(1234.0, 0, 1234.0);
        f(-12.34, 20, -12.34);
        f(12.0, 1, 10.0);
        f(25.0, 1, 30.0);
        f(2.5, 1, 3.0);
        f(-0.56, 1, -0.6);
        f(1234567.0, 3, 1230000.0);
        f(-1.234567, 4, -1.235);
        f(NAN, 10, NAN);
        f(*STALE_NAN, 10, *STALE_NAN)
    }

    #[test]
    fn test_positive_float_to_decimal() {
        let f = |f: f64, decimal_expected: i64, exponent_expected: i16| {
            let (decimal, exponent) = positive_float_to_decimal(f);
            assert_eq!(decimal, decimal_expected, "unexpected decimal for positiveFloatToDecimal({}); got {}; want {}", f, decimal, decimal_expected);
            assert_eq!(exponent, exponent_expected, "unexpected exponent for positiveFloatToDecimal({}); got {}; want {}",
                       f, exponent, exponent_expected);
        };

        f(0.0, 0, 1); // The exponent is 1 is OK here. See comment in positiveFloatToDecimal.
        f(1.0, 1, 0);
        f(30.0, 3, 1);
        f(12345678900000000_f64, 123456789, 8);
        f(12345678901234567_f64, 12345678901234568, 0);
        f(1234567890123456789_f64, 12345678901234567, 2);
        f(12345678901234567890_f64, 12345678901234567, 3);
        f(18446744073670737131_f64, 18446744073670737, 3);
        f(123456789012345678901_f64, 12345678901234568, 4);
        f((1_i64 << 53) as f64, 1<<53, 0);
        f((1_i64 << 54) as f64, 18014398509481984, 0);
        f((1_i64 << 55) as f64, 3602879701896396, 1);
        f((1_i64 << 62) as f64, 4611686018427387, 3);
        f((1_i64 << 63) as f64, 9223372036854775, 3);
        // Skip this test, since M1 returns 18446744073709551 instead of 18446744073709548
        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1653
        // f(1<<64, 18446744073709548, 3);
        f((1_i64 << 65) as f64, 368934881474191, 5);
        f((1_i64 << 66) as f64, 737869762948382, 5);
        f((1_i64 << 67) as f64, 1475739525896764, 5);

        f(0.1, 1, -1);
        f(123456789012345678e-5, 12345678901234568, -4);
        f(1234567890123456789e-10, 12345678901234568, -8);
        f(1234567890123456789e-14, 1234567890123, -8);
        f(1234567890123456789e-17, 12345678901234, -12);
        f(1234567890123456789e-20, 1234567890123, -14);

        f(0.000874957, 874957, -9);
        f(0.001130435, 1130435, -9);
        f(V_INF_POS as f64, 9223372036854775, 3);
        f(V_MAX as f64, 9223372036854775, 3);

        // Extreme cases. See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/1114
        f(2.964393875e-100, 2964393875, -109);
        f(2.964393875e-309, 2964393875, -318);
        f(2.964393875e-314, 296439387505, -325);
        f(2.964393875e-315, 2964393875047, -327);
        f(2.964393875e-320, 296439387505, -331);
        f(2.964393875e-324, 494065645841, -335);
        f(2.964393875e-325, 0, 1);

        f(2.964393875e+307, 2964393875, 298);
        f(9.964393875e+307, 9964393875, 298);
        f(1.064393875e+308, 1064393875, 299);
        f(1.797393875e+308, 1797393875, 299);
    }

    #[test]
    fn test_append_decimal_to_float() {
        check_append_decimal_to_float(&[], 0, &[]);
        check_append_decimal_to_float(&[0], 0, &[0.0]);
        check_append_decimal_to_float(&[0], 10, &[0.0]);
        check_append_decimal_to_float(&[0], -10, &[0.0]);
        check_append_decimal_to_float(&[-1, -10, 0, 100], 2, &[-1e2, -1e3, 0.0, 1e4]);
        check_append_decimal_to_float(&[-1, -10, 0, 100], -2, &[-1e-2, -1e-1, 0.0, 1.0]);
        check_append_decimal_to_float(&[874957, 1130435], -5, &[8.74957, 1.130435e1]);
        check_append_decimal_to_float(&[874957, 1130435], -6, &[8.74957e-1, 1.130435]);
        check_append_decimal_to_float(&[874957, 1130435], -7, &[8.74957e-2, 1.130435e-1]);
        check_append_decimal_to_float(&[874957, 1130435], -8, &[8.74957e-3, 1.130435e-2]);
        check_append_decimal_to_float(&[874957, 1130435], -9, &[8.74957e-4, 1.130435e-3]);
        check_append_decimal_to_float(&[874957, 1130435], -10, &[8.74957e-5, 1.130435e-4]);
        check_append_decimal_to_float(&[874957, 1130435], -11, &[8.74957e-6, 1.130435e-5]);
        check_append_decimal_to_float(&[874957, 1130435], -12, &[8.74957e-7, 1.130435e-6]);
        check_append_decimal_to_float(&[874957, 1130435], -13, &[8.74957e-8, 1.130435e-7]);
        check_append_decimal_to_float(&[V_MAX, V_MIN, 1, 2], 4, &[V_MAX * 1e4, V_MIN * 1e4, 1e4, 2e4]);
        check_append_decimal_to_float(&[V_MAX, V_MIN, 1, 2], -4, &[V_MAX * 1e-4, V_MIN * 1e-4, 1e-4, 2e-4]);
        check_append_decimal_to_float(&[V_INF_POS, V_INF_NEG, 1, 2], 0, &[infPos, infNeg, 1, 2]);
        check_append_decimal_to_float(&[V_INF_POS, V_INF_NEG, 1, 2], 4, &[infPos, infNeg, 1e4, 2e4]);
        check_append_decimal_to_float(&[V_INF_POS, V_INF_NEG, 1, 2], -4, &[infPos, infNeg, 1e-4, 2e-4]);
        check_append_decimal_to_float(&[1234, vStaleNaN, 1, 2], 0, &[1234, StaleNaN, 1, 2]);
        check_append_decimal_to_float(&[V_INF_POS, vStaleNaN, V_MIN, 2], 4, &[infPos, StaleNaN, V_MIN * 1e4, 2e4]);
        check_append_decimal_to_float(&[V_INF_POS, vStaleNaN, V_MIN, 2], -4, &[infPos, StaleNaN, V_MIN * 1e-4, 2e-4]);
    }

    fn check_append_decimal_to_float(va: &[i64], e: i16, expected: &[f64]) {
        let mut dst: Vec<f64> = Vec::with_capacity(va.len());
        append_decimal_to_float(&mut dst, va, e);
        assert!(equal_values(&dst, expected), "unexpected f for va={:?}, e={:?}: got\n{:?}; expecting\n{:?}",
                va, e, &dst, expected);

        let prefix = [1_f64, 2.0, 3.0, 4.0];
        let mut dst: Vec<f64> = Vec::from(prefix);
        append_decimal_to_float(&mut dst, va, e);
        let new_prefix = &dst[0 .. prefix.len()];
        let suffix = &dst[prefix.len() .. ];
        assert!(equal_values(new_prefix, &prefix),
                "unexpected prefix for va={:?}, e={}; got\n{:?}; expecting\n{:?}", va, e, new_prefix, prefix);

        assert!(equal_values(suffix, expected),
                "unexpected prefixed f for va={:?}, e={}: got\n{:?}; expecting\n{:?}", va, e, suffix, expected);
    }

    fn equal_values(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false
        }
        for (i, va) in a.iter().enumerate() {
            let vb = b[i];
            if va.to_bits() != vb.to_bits() {
                return false
            }
        }
        return true
    }

    #[test]
    fn test_calibrate_scale() {
        check_calibrate_scale(&[], &[], 0, 0, &[], &[], 0);
        check_calibrate_scale(&[0], &[0], 0, 0, &[0], &[0], 0);
        check_calibrate_scale(&[0], &[1], 0, 0, &[0], &[1], 0);
        check_calibrate_scale(&[1, 0, 2], &[5, -3], 0, 1, &[1, 0, 2], &[50, -30], 0);
        check_calibrate_scale(&[-1, 2], &[5, 6, 3], 2, -1, &[-1000, 2000], &[5, 6, 3], -1);
        check_calibrate_scale(&[123, -456, 94], &[-9, 4, -3, 45], -3, -3, &[123, -456, 94], &[-9, 4, -3, 45], -3);
        check_calibrate_scale(&[1e18, 1, 0], &[3, 456], 0, -2, &[1e18, 1, 0], &[0, 4], 0);
        check_calibrate_scale(&[12345, 678], &[12, -1e17, -3], -3, 0, &[123, 6], &[120, -1e18, -30], -1);
        check_calibrate_scale(&[1, 2], &[], 12, 34, &[1, 2], &[], 12);
        check_calibrate_scale(&[], &[3, 1], 12, 34, &[], &[3, 1], 34);
        check_calibrate_scale(&[923], &[2, 3], 100, -100, &[923e15], &[0, 0], 85);
        check_calibrate_scale(&[923], &[2, 3], -100, 100, &[0], &[2e18, 3e18], 82);
        check_calibrate_scale(&[123, 456, 789, 135], &[], -12, -10, &[123, 456, 789, 135], &[], -12);
        check_calibrate_scale(&[123, 456, 789, 135], &[], -10, -12, &[123, 456, 789, 135], &[], -10);

        check_calibrate_scale(&[V_INF_POS, 1200], &[500, 100], 0, 0, &[V_INF_POS, 1200], &[500, 100], 0);
        check_calibrate_scale(&[V_INF_POS, 1200], &[500, 100], 0, 2, &[V_INF_POS, 1200], &[500e2, 100e2], 0);
        check_calibrate_scale(&[V_INF_POS, 1200], &[500, 100], 0, -2, &[V_INF_POS, 12e4], &[500, 100], -2);
        check_calibrate_scale(&[V_INF_POS, 1200], &[3500, 100], 0, -3, &[V_INF_POS, 12e5], &[3500, 100], -3);
        check_calibrate_scale(&[V_INF_POS, 1200], &[35, 1], 0, 40, &[V_INF_POS, 0], &[35e17 as i64], 1e17 as i64]], 23);
        check_calibrate_scale(&[V_INF_POS, 1200], &[35, 1], 40, 0, &[V_INF_POS, 12e17 as i64], &[0, 0], 25);
        check_calibrate_scale(&[V_INF_NEG, 1200], &[35, 1], 35, -5, &[V_INF_NEG, 12e17 as i64]], &[0, 0], 20);
        check_calibrate_scale(&[V_MAX, V_MIN, 123], &[100], 0, 3, &[V_MAX, V_MIN, 123], &[100_000], 0);
        check_calibrate_scale(&[V_MAX, V_MIN, 123], &[100], 3, 0, &[V_MAX, V_MIN, 123], &[0], 3);
        check_calibrate_scale(&[V_MAX, V_MIN, 123], &[100], 0, 30, &[92233, -92233, 0], &[100e16], 14);
        check_calibrate_scale(&[vStaleNaN, V_MIN, 123], &[100], 0, 30, &[vStaleNaN, -92233, 0], &[100e16], 14);

        // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/805
        check_calibrate_scale(&[123], &[V_INF_POS], 0, 0, &[123], &[V_INF_POS], 0);
        check_calibrate_scale(&[123, V_INF_POS], &[V_INF_NEG], 0, 0, &[123, V_INF_POS], &[V_INF_NEG], 0);
        check_calibrate_scale(&[123, V_INF_POS, V_INF_NEG], &[456], 0, 0, &[123, V_INF_POS, V_INF_NEG], &[456], 0);
        check_calibrate_scale(&[123, V_INF_POS, V_INF_NEG, 456], &[], 0, 0, &[123, V_INF_POS, V_INF_NEG, 456], &[], 0);
        check_calibrate_scale(&[123, V_INF_POS], &[V_INF_NEG, 456], 0, 0, &[123, V_INF_POS], &[V_INF_NEG, 456], 0);
        check_calibrate_scale(&[123, V_INF_POS], &[V_INF_NEG, 456], 0, 10, &[123, V_INF_POS], &[V_INF_NEG, 456e10], 0)
    }

    fn check_calibrate_scale(a: &[i64], b: &[i64], ae: i16, be: i16, aExpected: &[i64], bExpected: &[i64],
                             eExpected: i16) {
        let mut aCopy = a.clone();
        let mut bCopy = b.clone();
        let e = calibrate_scale(&mut aCopy, ae, &mut bCopy, be);
        assert_eq!(e, eExpected, "unexpected e for a={:?}, b={:?}, ae={}, be={}; got {}; expecting {:?}", a, b, ae, be, e, eExpected);
        assert_eq!(aCopy, aExpected,
                   "unexpected a for b={:?}, ae={}, be={}; got\n{:?}; expecting\n{:?}", b, ae, be, aCopy, aExpected);

        assert_eq!(bCopy, bExpected,
                   "unexpected b for a={:?}, ae={}, be={}; got\n{:?}; expecting\n{:?}", a, ae, be, bCopy, bExpected);

        // Try reverse args.
        aCopy = a.clone();
        bCopy = b.clone();
        let e = calibrate_scale(&mut bCopy, be, &mut aCopy, ae);
        assert_eq!(e, eExpected, "reverse: unexpected e for a={:?}, b={:?}, ae={}, be={}; got {}; expecting {:?}",
                   a, b, ae, be, e, eExpected);

        assert_eq!(aCopy, aExpected,
                   "reverse: unexpected a for b={:?}, ae={}, be={}; got\n{:?}; expecting\n{:?}", b, ae, be, aCopy, aExpected);

        assert_eq!(bCopy, bExpected,
                   "reverse: unexpected b for a={:?}, ae={}, be={}; got\n{:?}; expecting\n{:?}", a, ae, be, bCopy, bExpected);
    }

    #[test]
    fn test_max_up_exponent() {
        let f = |v: i64, e_expected: i16| {
            let e = max_up_exponent(v);
            assert_eq!(e, e_expected, "unexpected e for v={}; got {}; expecting {}", v, e, e_expected);
        };

        f(V_INF_POS, 1024);
        f(V_INF_NEG, 1024);
        f(*STALE_NAN, 1024);
        f(V_MIN, 0);
        f(V_MAX, 0);
        f(0, 1024);
        f(1, 18);
        f(12, 17);
        f(123, 16);
        f(1234, 15);
        f(12345, 14);
        f(123456, 13);
        f(1234567, 12);
        f(12345678, 11);
        f(123456789, 10);
        f(1234567890, 9);
        f(12345678901, 8);
        f(123456789012, 7);
        f(1234567890123, 6);
        f(12345678901234, 5);
        f(123456789012345, 4);
        f(1234567890123456, 3);
        f(12345678901234567, 2);
        f(123456789012345678, 1);
        f(1234567890123456789, 0);
        f(923456789012345678, 0);
        f(92345678901234567, 1);
        f(9234567890123456, 2);
        f(923456789012345, 3);
        f(92345678901234, 4);
        f(9234567890123, 5);
        f(923456789012, 6);
        f(92345678901, 7);
        f(9234567890, 8);
        f(923456789, 9);
        f(92345678, 10);
        f(9234567, 11);
        f(923456, 12);
        f(92345, 13);
        f(9234, 14);
        f(923, 15);
        f(92, 17);
        f(9, 18);

        f(-1, 18);
        f(-12, 17);
        f(-123, 16);
        f(-1234, 15);
        f(-12345, 14);
        f(-123456, 13);
        f(-1234567, 12);
        f(-12345678, 11);
        f(-123456789, 10);
        f(-1234567890, 9);
        f(-12345678901, 8);
        f(-123456789012, 7);
        f(-1234567890123, 6);
        f(-12345678901234, 5);
        f(-123456789012345, 4);
        f(-1234567890123456, 3);
        f(-12345678901234567, 2);
        f(-123456789012345678, 1);
        f(-1234567890123456789, 0);
        f(-923456789012345678, 0);
        f(-92345678901234567, 1);
        f(-9234567890123456, 2);
        f(-923456789012345, 3);
        f(-92345678901234, 4);
        f(-9234567890123, 5);
        f(-923456789012, 6);
        f(-92345678901, 7);
        f(-9234567890, 8);
        f(-923456789, 9);
        f(-92345678, 10);
        f(-9234567, 11);
        f(-923456, 12);
        f(-92345, 13);
        f(-9234, 14);
        f(-923, 15);
        f(-92, 17);
        f(-9, 18);
    }

    #[test]
    fn test_append_float_to_decimal() {
        // no-op
        check_append_to_decimal(&[], &[], 0);
        check_append_to_decimal(&[0], &[0], 0);
        check_append_to_decimal(&[infPos, infNeg, 123], &[V_INF_POS, V_INF_NEG, 123], 0);
        check_append_to_decimal(&[infPos, infNeg, 123, 1e-4, 1e32], &[V_INF_POS, V_INF_NEG, 0, 0, 1000000000000000000], 14);
        check_append_to_decimal(&[StaleNaN, infNeg, 123, 1e-4, 1e32], &[*STALE_NAN, V_INF_NEG, 0, 0, 1000000000000000000_f64], 14);
        check_append_to_decimal(&[0, -0, 1, -1, 12345678, -123456789], &[0, 0, 1, -1, 12345678, -123456789], 0);

        // upExp
        check_append_to_decimal(&[-24, 0.0, 4.123, 0.3], &[-24000, 0, 4123, 300], -3);
        check_append_to_decimal(&[0, 10.23456789, 1e2, 1e-3, 1e-4], &[0, 1023456789, 1e10, 1e5, 1e4], -8);

        // downExp
        check_append_to_decimal(&[3e17, 7e-2, 5e-7, 45.0, 7e-1], &[3e18, 0, 0, 450, 7], -1);
        check_append_to_decimal(&[3e18, 1.0, 0.1, 13.0], &[3e18, 1.0, 0.0, 13.0], 0);
    }

    fn check_append_to_decimal(fa: &[f64], daExpected: &[i64], eExpected: i16) {
        let mut da: Vec<i64> = Vec::with_capacity(daExpected.len());
        let e = append_float_to_decimal(&mut da, fa);
        assert_eq!(e, eExpected, "unexpected e for fa={:?}; got {}; expecting {}", fa, e, eExpected);
        assert_eq!(da, daExpected,
                   "unexpected da for fa={:?}; got\n{:?}; expecting\n{:?}", fa, da, daExpected);

        let daPrefix = [1, 2, 3];
        let mut da: Vec<i64> = Vec::from(daPrefix);
        let e = append_float_to_decimal(&mut da, fa);
        let new_prefix = &da[0 .. daPrefix.len()];
        let suffix = &da[daPrefix.len() .. ];
        assert_eq!(e, eExpected, "unexpected e for fa={:?}; got {}; expecting {}", fa, e, eExpected);
        assert_eq!(new_prefix, daPrefix,
                   "unexpected daPrefix for fa={:?}; got\n{:?}; expecting\n{:?}", fa, new_prefix, daPrefix);
        assert_eq!(suffix, daExpected,
                   "unexpected da for fa={:?}; got\n{:?}; expecting\n{:?}", fa, suffix, daExpected);
    }

    #[test]
    fn test_float_to_decimal() {
        let f = |f: f64, v_expected: i64, e_expected: i16| {
            let (v, e) = from_float(f);
            assert_eq!(v, v_expected, "unexpected v for f={}; got {}; expecting {}", f, v, v_expected);
            assert_eq!(e, e_expected, "unexpected e for f={}; got {}; expecting {}", f, e, e_expected);
        };

        f(0.0, 0, 0);
        f(1.0, 1, 0);
        f(-1.0, -1, 0);
        f(0.9, 9, -1);
        f(0.99, 99, -2);
        f(9.0, 9, 0);
        f(99.0, 99, 0);
        f(20.0, 2, 1);
        f(100.0, 1, 2);
        f(3000.0, 3, 3);

        f(0.123, 123, -3);
        f(-0.123, -123, -3);
        f(1.2345, 12345, -4);
        f(-1.2345, -12345, -4);
        f(12000.0, 12, 3);
        f(-12000.0, -12, 3);
        f(1e-30, 1, -30);
        f(-1e-30, -1, -30);
        f(1e-260, 1, -260);
        f(-1e-260, -1, -260);
        f(321e260, 321, 260);
        f(-321e260, -321, 260);
        f(1234567890123.0, 1234567890123, 0);
        f(-1234567890123.0, -1234567890123, 0);
        f(123e5, 123, 5);
        f(15e18, 15, 18);

        f(f64::INFINITY, V_INF_POS, 0);
        f(f64::NEG_INFINITY, V_INF_NEG, 0);
        f(StaleNaN, vStaleNaN, 0);
        f(V_INF_POS, 9223372036854775, 3);
        f(V_INF_NEG, -9223372036854775, 3);
        f(V_MAX, 9223372036854775, 3);
        f(V_MIN, -9223372036854775, 3);
        f(1<<63-1, 9223372036854775, 3);
        f(-1<<63, -9223372036854775, 3);

        // Test precision loss due to conversionPrecision.
        f(0.1234567890123456, 12345678901234, -14);
        f(-123456.7890123456, -12345678901234, -8);
    }

    #[test]
    fn test_float_to_decimal_roundtrip() {
        let f = |f: f64| {
            let (v, e) = from_float(f);
            let fNew = to_float(v, e);
            if !equal_float(f, fNew) {
                panic!("unexpected fNew for v={}, e={}; got {}; expecting {}", v, e, fNew, f);
            }

            let (v, e) = from_float(-f);
            let fNew = to_float(v, e);
            if !equal_float(-f, fNew) {
                panic!("unexpected fNew for v={}, e={}; got {}; expecting {}", v, e, fNew, -f);
            }
        };

        f(0.0);
        f(1.0);
        f(0.123);
        f(1.2345);
        f(12000.0);
        f(1e-30);
        f(1e-260);
        f(321e260);
        f(1234567890123.0);
        f(12.34567890125);
        f(1234567.8901256789);
        f(15e18);
        f(0.000874957);
        f(0.001130435);

        f(2933434554455e245);
        f(3439234258934e-245);
        f(V_INF_POS as f64);
        f(V_INF_NEG as f64);
        f(infPos);
        f(infNeg);
        f(V_MAX);
        f(V_MIN);
        f(vStaleNaN);

        for i in 0 .. 1_0000 {
            let v = rand.NormFloat64();
            f(v);
            f(v * 1e-6);
            f(v * 1e6);

            f(round_float(v, 20));
            f(round_float(v, 10));
            f(round_float(v, 5));
            f(round_float(v, 0));
            f(round_float(v, -5));
            f(round_float(v, -10));
            f(round_float(v, -20))
        }
    }

    fn round_float(f: f64, exp: i32) -> f64 {
        let f = f * (-1 * exp).pow(10) as f64;
        return f.trunc() * exp.pow(10) as f64;
    }

    fn equal_float(f1: f64, f2: f64) -> bool {
        if f1.is_finite() {
            if !f2.is_finite() {
                return false;
            }
            let eps = (f1 - f2).abs();
            return eps == 0.0 || eps * CONVERSION_PRECISION < f1.abs() + f2.abs();
        }
        if f1 == f64::INFINITY {
            return f2 == f64::INFINITY
        }
        return f2 == f64::NEG_INFINITY
    }
}
