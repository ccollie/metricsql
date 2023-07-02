#[cfg(test)]
mod tests {
    use rand::{Rng, thread_rng};

    use crate::encoding::unmarshal_int64_nearest_delta;
    use crate::marshal_int64_nearest_delta;
    use crate::tests::utils::{check_precision_bits, get_rand_normal};

    #[test]
    fn test_marshal_int64nearest_delta() {
        check_marshal_unmarshal_i64(&[0], 4, 0, "");
        check_marshal_unmarshal_i64(&[0, 0], 4, 0, "00");
        check_marshal_unmarshal_i64(&[1, -3], 4, 1, "07");
        check_marshal_unmarshal_i64(&[255, 255], 4, 255, "00");
        check_marshal_unmarshal_i64(&[0, 1, 2, 3, 4, 5], 4, 0, "0202020202");
        check_marshal_unmarshal_i64(&[5, 4, 3, 2, 1, 0], 1, 5, "0003000301");
        check_marshal_unmarshal_i64(&[5, 4, 3, 2, 1, 0], 2, 5, "0003010101");
        check_marshal_unmarshal_i64(&[5, 4, 3, 2, 1, 0], 3, 5, "0101010101");
        check_marshal_unmarshal_i64(&[5, 4, 3, 2, 1, 0], 4, 5, "0101010101");

        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 1, -500, "00000000");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 2, -500, "0000ff0300");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 3, -500, "00ff01ff01ff01");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 4, -500, "7fff017fff01");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 5, -500, "bf01bf01bf01bf01");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 6, -500, "bf01bf01bf01bf01");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 7, -500, "bf01cf01bf01af01");
        check_marshal_unmarshal_i64(&[-500, -600, -700, -800, -890], 8, -500, "c701c701c701af01");
    }

    fn check_marshal_unmarshal_i64(
        va: &[i64],
        precision_bits: u8,
        first_value_expected: i64,
        b_expected: &str,
    ) {
        let mut b: Vec<u8> = vec![];
        let first_value = marshal_int64_nearest_delta(&mut b, va, precision_bits)
            .expect("marshal_int64_nearest_delta");
        assert_eq!(
            first_value, first_value_expected,
            "unexpected first_value for va={:?}, precision_bits={}; got {}; want {}",
            va, precision_bits, first_value, first_value_expected
        );

        assert_eq!(
            &b,
            b_expected.as_bytes(),
            "invalid marshaled data for va={:?}, precision_bits={}; got\n{:?}; expecting\n{}",
            va,
            precision_bits,
            b,
            b_expected
        );

        let prefix = b"foobar".as_slice();
        let mut b: Vec<u8> = Vec::from(prefix);
        let first_value = marshal_int64_nearest_delta(&mut b, va, precision_bits)
            .expect("marshal_int64_nearest_delta");
        assert_eq!(
            first_value, first_value_expected,
            "unexpected first_value for va={:?}, precision_bits={}; got {}; want {}",
            va, precision_bits, first_value, first_value_expected
        );

        let new_prefix = &b[0..prefix.len()];
        assert_eq!(
            new_prefix, prefix,
            "invalid prefix for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
            va, precision_bits, new_prefix, prefix
        );

        let suffix = &b[prefix.len()..];
        assert_eq!(suffix, b_expected.as_bytes(),
                   "invalid marshaled prefixed data for va={:?}, precision_bits={}; got\n{:?}; expecting\n{}",
                   va, precision_bits, suffix, b_expected)
    }

    #[test]
    fn test_marshal_unmarshal_int64nearest_delta() {
        let (val1, val2, val3, val4, val5) = (
            -5e12 as i64,
            -6e12 as i64,
            -7e12 as i64,
            -8e12 as i64,
            -8.9e12 as i64,
        );
        let v = -5.6e12 as i64;
        check_int64_nearest_delta(&[0], 4);
        check_int64_nearest_delta(&[0, 0], 4);
        check_int64_nearest_delta(&[1, -3], 4);
        check_int64_nearest_delta(&[255, 255], 4);
        check_int64_nearest_delta(&[0, 1, 2, 3, 4, 5], 4);
        check_int64_nearest_delta(&[5, 4, 3, 2, 1, 0], 4);
        check_int64_nearest_delta(&[val1, val2, val3, val4, val5], 1);
        check_int64_nearest_delta(&[val1, val2, val3, val4, val5], 2);
        check_int64_nearest_delta(&[val1, val2, val3, val4, val5], 3);
        check_int64_nearest_delta(&[val1, v, val3, val4, val5], 4);

        let mut rng = thread_rng();

        // Verify constant encoding.
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            va.push(9876543210123);
        }
        check_int64_nearest_delta(&va, 4);
        check_int64_nearest_delta(&va, 63);

        // Verify encoding for monotonically incremented va.
        let mut v: i64 = -35;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v += 8;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);
        check_int64_nearest_delta(&va, 63);

        // Verify encoding for monotonically decremented va.
        let mut v = 793;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v -= 16;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);
        check_int64_nearest_delta(&va, 63);

        // Verify encoding for quadratically incremented va.
        v = -1234567;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for i in 0..1024 {
            v += 32 + i as i64;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);

        // Verify encoding for decremented va with norm-float noise.
        v = 787933;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v -= 25 + (get_rand_normal() * 2_f64) as i64;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);

        // Verify encoding for incremented va with random noise.
        v = 943854;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v += 30 + rng.gen_range(0..5);
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);

        // Verify encoding for constant va with norm-float noise.
        v = -12345;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v += (get_rand_normal() * 10_f64) as i64;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4);

        // Verify encoding for constant va with random noise.
        v = -12345;
        let mut va: Vec<i64> = Vec::with_capacity(1024);
        for _ in 0..1024 {
            v += rng.gen_range(0..15) - 1;
            va.push(v);
        }
        check_int64_nearest_delta(&va, 4)
    }

    fn check_int64_nearest_delta(va: &[i64], precision_bits: u8) {
        let mut b: Vec<u8> = vec![];

        let first_value = marshal_int64_nearest_delta(&mut b, va, precision_bits)
            .expect("marshal_int64_nearest_delta");

        let mut va_new: Vec<i64> = vec![];
        match unmarshal_int64_nearest_delta(&mut va_new, &b, first_value, va.len()) {
            Err(err) => {
                panic!(
                    "cannot unmarshal data for va={:?}, precision_bits={} from b={:?}: {:?}",
                    va, precision_bits, b, err
                )
            }
            Ok(_) => {
                if let Err(err) = check_precision_bits(&va_new, va, precision_bits) {
                    panic!("too small precision_bits for va={:?}, precision_bits={}: {:?}, va_new=\n{:?}",
                           va, precision_bits, err, va_new)
                }
            }
        }

        let va_prefix = [1, 2, 3, 4];
        let mut va_new: Vec<i64> = Vec::from(va_prefix);
        match unmarshal_int64_nearest_delta(&mut va_new, &b, first_value, va.len()) {
            Ok(_) => {}
            Err(err) => {
                panic!("cannot unmarshal prefixed data for va={:?}, precision_bits={} from b={:?}: {:?}",
                       va, precision_bits, b, err);
            }
        }

        let prefix = &va_new[0..va_prefix.len()];
        assert_eq!(
            prefix, va_prefix,
            "unexpected prefix for va={:?}, precision_bits={}: got\n{:?}; expecting\n{:?}",
            va, precision_bits, prefix, va_prefix
        );

        let suffix = &va_new[va_prefix.len()..];
        if let Err(err) = check_precision_bits(suffix, va, precision_bits) {
            panic!(
                "too small precision_bits for va={:?}, precision_bits={}: {:?}, va_new=\n{:?}",
                va, precision_bits, err, suffix
            )
        }

        if let Err(err) = check_precision_bits(&va_new, va, precision_bits) {
            panic!(
                "too small precision_bits for va={:?}, precision_bits={}: {:?}, va_new=\n{:?}",
                va, precision_bits, err, va_new
            )
        }
    }
}
