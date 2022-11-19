#[cfg(test)]
mod tests {
    use crate::encoding::{is_const};
    use crate::{ensure_non_decreasing_sequence, is_delta_const, is_gauge,
                marshal_timestamps, marshal_values, MarshalType,
                unmarshal_timestamps, unmarshal_values};

    use crate::tests::utils::{check_precision_bits, ensure_marshal_unmarshal_int64_array, get_rand_normal};

    #[test]
    fn test_is_const() {
        let f = |a: &[i64], ok_expected: bool| {
            let ok = is_const(a);
            assert_eq!(ok, ok_expected, "unexpected is_const for a={:?}; got {}; want {}", a, ok, ok_expected);
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
            assert_eq!(ok, ok_expected, "unexpected isDeltaConst for a={:?}; got {}; want {}", a, ok, ok_expected);
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
            assert_eq!(ok, ok_expected, "unexpected result for is_gauge({:?}); got {}; expecting {}",
                       values, ok, ok_expected);
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
        assert_eq!(values, expected, "unexpected a; got\n{:?}; expecting\n{:?}", values, expected);
    }


    #[test]
    fn test_marshal_unmarshal_timestamps() {
        const PRECISION_BITS: u8 = 3;

        let mut timestamps: Vec<i64> = vec![];
        let mut timestamps2: Vec<i64> = vec![];

        let mut v: i64 = 0;

        for _ in 0 .. 8*1024 {
            v += (30e3 * get_rand_normal() * 5e2) as i64;
            timestamps.push(v);
        }

        let mut result: Vec<u8> = vec![];

        let (mt, first_timestamp) = marshal_timestamps(&mut result, &timestamps, PRECISION_BITS)
            .expect("marshal_timestamps");

        unmarshal_timestamps(&mut timestamps2, &result, mt, first_timestamp, timestamps.len())
            .expect("unmarshal_timestamps");

        match check_precision_bits(&timestamps, &timestamps2, PRECISION_BITS) {
            Err(err) => panic!("too low precision for timestamps: {:?}", err),
            _ => {}
        }
    }

    #[test]
    fn test_marshal_unmarshal_values() {
        const PRECISION_BITS: u8 = 3;

        let mut values: Vec<i64> = vec![];
        let mut v: i64 = 0;
        
        for _ in 0 .. 8*1024 {
            v += (get_rand_normal() * 1e2) as i64;
            values.push(v)
        }
        let mut result: Vec<u8> = vec![];
        let mut values2: Vec<i64> = vec![];
        let (mt, first_value) = marshal_values(&mut result, &values, PRECISION_BITS)
            .expect("marshal_values");
        
        unmarshal_values(&mut values2, &result, mt, first_value, values.len())
            .expect("unmarshal values");
        
        assert_eq!(values2, values);

        match check_precision_bits(&values, &values2, PRECISION_BITS) {
            Err(err) => panic!("too low precision for values: {:?}", err),
            _ => {}
        }
    }

    #[test]
    fn test_marshal_unmarshal_int64array_generic() {
        use MarshalType::*;

        ensure_marshal_unmarshal_int64_array(&[1, 20, 234], 4, NearestDelta2);
        ensure_marshal_unmarshal_int64_array(&[1, 20, -2345, 678934, 342], 4, NearestDelta);
        ensure_marshal_unmarshal_int64_array(&[1, 20, 2345, 6789, 12342], 4, NearestDelta2);

        // Constant encoding
        ensure_marshal_unmarshal_int64_array(&[1], 4, Const);
        ensure_marshal_unmarshal_int64_array(&[1, 2], 4, DeltaConst);
        ensure_marshal_unmarshal_int64_array(&[-1, 0, 1, 2, 3, 4, 5], 4, DeltaConst);
        ensure_marshal_unmarshal_int64_array(&[-10, -1, 8, 17, 26], 4, DeltaConst);
        ensure_marshal_unmarshal_int64_array(&[0, 0, 0, 0, 0, 0], 4, Const);
        ensure_marshal_unmarshal_int64_array(&[100, 100, 100, 100], 4, Const);
    }

}