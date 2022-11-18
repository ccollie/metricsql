#[cfg(test)]
mod tests {
    use rand::Rng;
    use crate::encoding::nearest_delta;
    use crate::encoding::nearest_delta2::{marshal_int64_nearest_delta2, unmarshal_int64_nearest_delta2};
    use crate::{get_trailing_zeros, rand_nextf64};
    use crate::tests::utils::check_precision_bits;

    #[test]
    fn test_marshal_int64nearest_delta2() {
        check_marshal_i64_nearest_delta2(&[0, 0], 4, 0, "00");
        check_marshal_i64_nearest_delta2(&[1, -3], 4, 1, "07");
        check_marshal_i64_nearest_delta2(&[255, 255], 4, 255, "00");
        check_marshal_i64_nearest_delta2(&[0, 1, 2, 3, 4, 5], 4, 0, "0200000000");
        check_marshal_i64_nearest_delta2(&[5, 4, 3, 2, 1, 0], 4, 5, "0100000000");

        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 1, -5000, "cf0f000000");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 2, -5000, "cf0f000000");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 3, -5000, "cf0f000000");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 4, -5000, "cf0f00008001");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 5, -5000, "cf0f0000c001");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 6, -5000, "cf0f0000c001");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 7, -5000, "cf0f0000c001");
        check_marshal_i64_nearest_delta2(&[-5000, -6000, -7000, -8000, -8_900_000_000_000], 8, -5000, "cf0f0000c801");
    }

    fn check_marshal_i64_nearest_delta2(va: &[i64], precision_bits: u8, first_value_expected: i64, b_expected: &str) {
        let mut b: Vec<u8> = vec![];

        let first_value = marshal_int64_nearest_delta2(&mut b, va, precision_bits).expect("marshal int64 nearest delta1");
        assert_eq!(first_value, first_value_expected,
                   "unexpected first_value for va={:?}, precision_bits={}; got {}; want {}", va, precision_bits, first_value, first_value_expected);

        let b_str = String::from_utf8(b.clone()).expect("convert to utf8");
        assert_eq!(b_str, b_expected,
                   "invalid marshaled data for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
                   va, precision_bits, b, b_expected);

        let prefix = b"foobar".as_slice();
        b.clear();
        b.extend_from_slice(prefix);

        let first_value = marshal_int64_nearest_delta2(&mut b, va, precision_bits).expect("marshal i64 nearest delta2");
        assert_eq!(first_value, first_value_expected,
                   "unexpected first_value for va={:?}, precision_bits={}; got {}; want {}", va, precision_bits, first_value, first_value_expected);

        let b_str = String::from_utf8(b.clone()).expect("convert to utf8");
        assert_eq!(b_str.as_bytes(), prefix, "invalid prefix for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
                   va, precision_bits, &b[0.. prefix.len()], prefix);

        let suffix = String::from_utf8(b[prefix.len() .. ].to_vec()).expect("convert to utf-8");
        assert_eq!(suffix, b_expected,
                   "invalid marshaled prefixed data for va={:?}, precision_bits={}; got\n{}; expecting\n{}",
                   va, precision_bits, suffix, b_expected)
    }


    #[test]
    fn test_nearest_delta() {
        check_nearest_delta(0, 0, 1, 0, 0);
        check_nearest_delta(0, 0, 2, 0, 0);
        check_nearest_delta(0, 0, 3, 0, 0);
        check_nearest_delta(0, 0, 4, 0, 0);

        check_nearest_delta(100, 100, 4, 0, 2);
        check_nearest_delta(123456, 123456, 4, 0, 12);
        check_nearest_delta(-123456, -123456, 4, 0, 12);
        check_nearest_delta(9876543210, 9876543210, 4, 0, 29);

        check_nearest_delta(1, 2, 3, -1, 0);
        check_nearest_delta(2, 1, 3, 1, 0);
        check_nearest_delta(-1, -2, 3, 1, 0);
        check_nearest_delta(-2, -1, 3, -1, 0);

        check_nearest_delta(0, 1, 1, -1, 0);
        check_nearest_delta(1, 2, 1, -1, 0);
        check_nearest_delta(2, 3, 1, 0, 1);
        check_nearest_delta(1, 0, 1, 1, 0);
        check_nearest_delta(2, 1, 1, 0, 1);
        check_nearest_delta(2, 1, 2, 1, 0);
        check_nearest_delta(2, 1, 3, 1, 0);

        check_nearest_delta(0, -1, 1, 1, 0);
        check_nearest_delta(-1, -2, 1, 1, 0);
        check_nearest_delta(-2, -3, 1, 0, 1);
        check_nearest_delta(-1, 0, 1, -1, 0);
        check_nearest_delta(-2, -1, 1, 0, 1);
        check_nearest_delta(-2, -1, 2, -1, 0);
        check_nearest_delta(-2, -1, 3, -1, 0);

        check_nearest_delta(0, 2, 3, -2, 0);
        check_nearest_delta(3, 0, 3, 3, 0);
        check_nearest_delta(4, 0, 3, 4, 0);
        check_nearest_delta(5, 0, 3, 5, 0);
        check_nearest_delta(6, 0, 3, 6, 0);
        check_nearest_delta(0, 7, 3, -7, 0);
        check_nearest_delta(8, 0, 3, 8, 1);
        check_nearest_delta(9, 0, 3, 8, 1);
        check_nearest_delta(15, 0, 3, 14, 1);
        check_nearest_delta(16, 0, 3, 16, 2);
        check_nearest_delta(17, 0, 3, 16, 2);
        check_nearest_delta(18, 0, 3, 16, 2);
        check_nearest_delta(0, 59, 6, -59, 0);

        check_nearest_delta(128, 121, 1, 0, 7);
        check_nearest_delta(128, 121, 2, 0, 6);
        check_nearest_delta(128, 121, 3, 0, 5);
        check_nearest_delta(128, 121, 4, 0, 4);
        check_nearest_delta(128, 121, 5, 0, 3);
        check_nearest_delta(128, 121, 6, 4, 2);
        check_nearest_delta(128, 121, 7, 6, 1);
        check_nearest_delta(128, 121, 8, 7, 0);

        check_nearest_delta(32, 37, 1, 0, 5);
        check_nearest_delta(32, 37, 2, 0, 4);
        check_nearest_delta(32, 37, 3, 0, 3);
        check_nearest_delta(32, 37, 4, -4, 2);
        check_nearest_delta(32, 37, 5, -4, 1);
        check_nearest_delta(32, 37, 6, -5, 0);

        check_nearest_delta(-10, 20, 1, -24, 3);
        check_nearest_delta(-10, 20, 2, -28, 2);
        check_nearest_delta(-10, 20, 3, -30, 1);
        check_nearest_delta(-10, 20, 4, -30, 0);
        check_nearest_delta(-10, 21, 4, -31, 0);
        check_nearest_delta(-10, 21, 5, -31, 0);

        check_nearest_delta(10, -20, 1, 24, 3);
        check_nearest_delta(10, -20, 2, 28, 2);
        check_nearest_delta(10, -20, 3, 30, 1);
        check_nearest_delta(10, -20, 4, 30, 0);
        check_nearest_delta(10, -21, 4, 31, 0);
        check_nearest_delta(10, -21, 5, 31, 0);

        check_nearest_delta(1234e12 as i64 as i64, 1235e12 as i64 as i64, 1, 0, 50);
        check_nearest_delta(1234e12 as i64 as i64, 1235e12 as i64 as i64, 10, 0, 41);
        check_nearest_delta(1234e12 as i64 as i64, 1235e12 as i64 as i64, 35, -999999995904, 16);

        check_nearest_delta((1<<63)-1, 0, 1, (1<<63)-1, 2);
    }

    fn check_nearest_delta(next: i64, prev: i64, precision_bits: u8, d_expected: i64, trailing_zero_bits_expected: u8) {
        let tz = get_trailing_zeros(prev, precision_bits);
        let (d, trailing_zero_bits) = nearest_delta(next, prev, precision_bits, tz as u8);
        assert_eq!(d, d_expected, "unexpected d for next={}, prev={}, precision_bits={}; got {}; expecting {}", next, prev, precision_bits, d, d_expected);
        assert_eq!(trailing_zero_bits, trailing_zero_bits_expected,
                   "unexpected trailing_zero_bits for next={}, prev={}, precision_bits={}; got {}; expecting {}",
                   next, prev, precision_bits, trailing_zero_bits, trailing_zero_bits_expected);
    }


    #[test]
    fn test_marshal_unmarshal_int64nearest_delta2() {
        check_marshal_unmarshal_int64_nearest_delta2(&[0, 0], 4);
        check_marshal_unmarshal_int64_nearest_delta2(&[1, -3], 4);
        check_marshal_unmarshal_int64_nearest_delta2(&[255, 255], 4);
        check_marshal_unmarshal_int64_nearest_delta2(&[0, 1, 2, 3, 4, 5], 4);
        check_marshal_unmarshal_int64_nearest_delta2(&[5, 4, 3, 2, 1, 0], 4);
        check_marshal_unmarshal_int64_nearest_delta2(&[-5e12 as i64, -6e12 as i64, -7e12 as i64, -8e12 as i64, -8.9e12 as i64], 1);
        check_marshal_unmarshal_int64_nearest_delta2(&[-5e12 as i64, -6e12 as i64, -7e12 as i64, -8e12 as i64, -8.9e12 as i64], 2);
        check_marshal_unmarshal_int64_nearest_delta2(&[-5e12 as i64, -6e12 as i64, -7e12 as i64, -8e12 as i64, -8.9e12 as i64], 3);
        check_marshal_unmarshal_int64_nearest_delta2(&[-5e12 as i64, -6e12 as i64, -7e12 as i64, -8e12 as i64, -8.9e12 as i64], 4);

        // Verify constant encoding.
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            va.push(9876543210123)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);
        check_marshal_unmarshal_int64_nearest_delta2(&va, 63);

        // Verify encoding for monotonically incremented va.
        let mut v = -35_i64;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            v += 8;
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);
        check_marshal_unmarshal_int64_nearest_delta2(&va, 63);

        // Verify encoding for monotonically decremented va.
        let mut v = 793;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            v -= 16;
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);
        check_marshal_unmarshal_int64_nearest_delta2(&va, 63);

        // Verify encoding for quadratically incremented va.
        let mut v = -1234567;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for i in 0 .. 1024 {
            v += 32 + i as i64;
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);
        check_marshal_unmarshal_int64_nearest_delta2(&va, 63);

        let mut rng = rand::thread_rng();

        // Verify encoding for decremented va with norm-float noise.
        let mut v = 787933;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            let mut x = rand_nextf64() * 2_f64;
            if x > f64::MAX {
                x = i64::MAX as f64;
            }
            v -= 25 + (x as i64);
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);

        // Verify encoding for incremented va with random noise.
        let mut v = 943854;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            v += 30 + rng.gen_range(0..5);
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 4);

        // Verify encoding for constant va with norm-float noise.
        let mut v = -12345;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            // ??????
            v += 10_i64.checked_mul(rng.gen_range(1..i64::MAX)).unwrap_or(0); // todo: handle overflow
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 2);

        // Verify encoding for constant va with random noise.
        let mut v = -12345;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0 .. 1024 {
            v += rng.gen_range(0..15) - 1;
            va.push(v)
        }
        check_marshal_unmarshal_int64_nearest_delta2(&va, 3)
    }

    fn check_marshal_unmarshal_int64_nearest_delta2(va: &[i64], precision_bits: u8) {
        let mut src: Vec<u8> = vec![];
        let first_value = marshal_int64_nearest_delta2(&mut src, va, precision_bits).expect("marshal nearest delta 2");

        let mut dst: Vec<i64> = vec![];

        match unmarshal_int64_nearest_delta2(&mut dst, &src, first_value, va.len()) {
            Err(err) => {
                panic!("cannot unmarshal data for va={:?}, precision_bits={} from b={:?}: {:?}", va, precision_bits, src, err)
            },
            Ok(_) => {
                let va_new = dst;
                match check_precision_bits(&va_new, va, precision_bits) {
                    Err(err) => {
                        panic!("too small precision_bits for va={:?}, precision_bits={}: {}, vaNew=\n{:?}",
                               va, precision_bits, err, va_new);
                    },
                    Ok(_) => {
                        let va_prefix = [1_i64, 2, 3, 4];
                        let mut buf: Vec<i64> = va_prefix.to_vec();
                        match unmarshal_int64_nearest_delta2(&mut buf, &src, first_value, va.len()) {
                            Ok(v) => v,
                            Err(err) => {
                                panic!("cannot unmarshal prefixed data for va={:?}, precision_bits={} from b={:?}: {}", va, precision_bits, &src, err)
                            }
                        };

                        let prefix = &va_new[0 .. va_prefix.len()];
                        let suffix = &va_new[va_prefix.len() .. ];
                        assert_eq!(prefix, va_prefix,
                                   "unexpected prefix for va={:?}, precision_bits={}: got\n{:?}; expecting\n{:?}",
                                   va, precision_bits, prefix, va_prefix);
                        match check_precision_bits(suffix, va, precision_bits) {
                            Err(err) => {
                                panic!("too small precision_bits for prefixed va={:?}, precision_bits={}: {:?}, va_new=\n{:?}",
                                       va, precision_bits, err, suffix)
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    
}
