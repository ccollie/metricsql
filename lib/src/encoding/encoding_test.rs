#[cfg(test)]
mod tests {


    #[test]
    fn test_is_const() {
        let f = |a: &[i64], ok_expected: bool| {
            let ok = is_const(a);
            assert_eq!(ok, ok_expected, "unexpected isConst for a={}; got {}; want {}", a, ok, ok_expected);
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
            let ok = isDeltaConst(a);
            assert_eq!(ok, ok_expected, "unexpected isDeltaConst for a={}; got {}; want {}", a, ok, ok_expected);
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
        let f = |a: &[i64], ok_expected: bool| {
            let ok = is_gauge(a);
            assert_eq!(ok, ok_expected, "unexpected result for isGauge({}); got {}; expecting {}", a, ok, ok_expected);
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

    fn _ensure_non_decreasing_sequence(a: &[i64], vMin: i64, vMax: i64, a_expected: &[i64]) {
        EnsureNonDecreasingSequence(a, vMin, vMax);
        assert_eq(a, a_expected, "unexpected a; got\n{:?}; expecting\n{:?}", a, a_expected);
    }

    fn test_marshal_unmarshal_int64array(va: &[i64], precision_bits: u8, mt_expected: MarshalType) {
        b, mt, firstValue = marshal_int64_array(nil, va, precision_bits);
        if mt != mt_expected {
            panic!("unexpected MarshalType for va={:?}, precision_bits={}: got {}; expecting {}", va, precision_bits, mt, mt_expected)
        }
        vaNew = unmarshalInt64Array(nil, b, mt, firstValue, va.len());
        if err != nil {
            panic!("unexpected error when unmarshalling va={:?}, precision_bits={}: {}", va, precision_bits, err)
        }
        if vaNew == nil && va != nil {
            vaNew = &[];
        }
        match mt { 
            Lz4NearestDelta | Lz4NearestDelta2 | 
            NearestDelta | NearestDelta2 => {
                match check_precision_bits(va, vaNew, precisionBits) {
                    Err(err) => panic!("too low precision for vaNew: {:?}", err),
                    _ => {}
                }                
            }, 
            _ => {
                assert_eq!(va, vaNew,
                           "unexpected vaNew for va={:?}, precisionBits={}; got\n{}; expecting\n{:?}",
                           va, precisionBits, vaNew, va)
            }
        }

        let bPrefix = [1, 2, 3];
        bNew, mtNew, firstValueNew = marshalInt64Array(bPrefix, va, precisionBits);
        assert_eq!(firstValueNew, firstValue, 
                   "unexpected firstValue for prefixed va={:?}, precisionBits={}; got {}; want {}", 
                   va, precisionBits, firstValueNew, firstValue);

        let new_prefix = &bNew[0..bPrefix.len()];
        assert_eq!(new_prefix, bPrefix, "unexpected prefix for va={:?}, precisionBits={}; got\n{}; expecting\n{}",
            va, precisionBits, new_prefix, bPrefix);
        let suffix = &bNew[bPrefix.len()..];
        assert_eq!(suffix, &b,
            "unexpected b for prefixed va={:?}, precisionBits={}; got\n{}; expecting\n{}",
            va, precisionBits, suffix, b);
        assert_eq!(mtNew, mt, "unexpected mt for prefixed va={:?}, precisionBits={}; got {}; expecting {}", 
                   va, precisionBits, mtNew, mt);
    
        let vaPrefix = [4, 5, 6, 8];
        vaNew, err = unmarshalInt64Array(vaPrefix, b, mt, firstValue, va.len());
        if err != nil {
            panic!("unexpected error when unmarshaling prefixed va={:?}, precisionBits={}: {}", va, precisionBits, err)
        }
        let prefix = &vaNew[0 .. vaPrefix.len()];
        assert_eq!(prefix, &vaPrefix,
            "unexpected prefix for va={:?}, precisionBits={}; got\n{}; expecting\n{}",
                va, precisionBits, prefix, vaPrefix);

        match mt {
            MarshalTypeZSTDNearestDelta |
            MarshalTypeZSTDNearestDelta2 |
            MarshalTypeNearestDelta | MarshalTypeNearestDelta2 => {
                let suffix = &vaNew[vaPrefix.len() .. ];
                if err = check_precision_bits(&suffix, va, precisionBits); err != nil {
                    panic!("too low precision for prefixed vaNew: {}", err)
                }
            },
            _ => {
                default:
                if !reflect.DeepEqual(vaNew[vaPrefix.len():], va) {
                    panic!("unexpected prefixed vaNew for va={:?}, precisionBits={}; got\n{}; expecting\n{}", va, precisionBits, vaNew[vaPrefix.len():], va)
                }
            }
        }
    }

        fn get_rand_normal() -> f64 {
            thread_rng().sample::<f64,_>(StandardNormal)
        }

    #[test]
    fn test_marshal_unmarshal_timestamps() {
        const PRECISION_BITS: u8 = 3;

        let timestamps: Vec<i64> = vec![];

        let mut v: i64 = 0;

        for i in 0 .. 8*1024 {
            v += 30e3 * int64(rand.NormFloat64()*5e2);
            timestamps.push(v);
        }

        result, mt, firstTimestamp := marshal_timestamps(&mut result, timestamps, precisionBits);
        timestamps2, err := UnmarshalTimestamps(nil, result, mt, firstTimestamp, len(timestamps))
        if err != nil {
            panic!("cannot unmarshal timestamps: {}", err)
        }

        match check_precision_bits(timestamps, timestamps2, precisionBits) {
            Err(e) => format!("too low precision for timestamps: {:?}", err),
            _ => {}
        }
    }

    #[test]
    fn test_marshal_unmarshal_values() {
        const PRECISION_BITS: u8 = 3;

        let mut values: Vec<i64> = vec![];
        let mut v: i64 = 0;
        
        for i in 0 .. 8*1024 {
            v += int64(rand.NormFloat64() * 1e2);
            values.push(v)
        }
        
        result, mt, firstValue := marshal_values(nil, values, PRECISION_BITS);
        values2, err := unmarshal_values(nil, result, mt, firstValue, values.len());
        if err != nil {
            panic!("cannot unmarshal values: {}", err)
        }
        
        match check_precision_bits(values, values2, PRECISION_BITS) {
            Err(e) => format!("too low precision for values: {:?}", err),
            _ => {}
        }
    }

    fn test_marshal_unmarshal_int64array_generic() {
        _test_marshal_int64array_size(&[1, 20, 234], 4, MarshalTypeNearestDelta2);
        _test_marshal_int64array_size(&[1, 20, -2345, 678934, 342], 4, MarshalTypeNearestDelta);
        _test_marshal_int64array_size(&[1, 20, 2345, 6789, 12342], 4, MarshalTypeNearestDelta2);

        // Constant encoding
        _test_marshal_int64array_size(&[1], 4, MarshalTypeConst);
        _test_marshal_int64array_size(&[1, 2], 4, MarshalTypeDeltaConst);
        _test_marshal_int64array_size(&[-1, 0, 1, 2, 3, 4, 5], 4, MarshalTypeDeltaConst);
        _test_marshal_int64array_size(&[-10, -1, 8, 17, 26], 4, MarshalTypeDeltaConst);
        _test_marshal_int64array_size(&[0, 0, 0, 0, 0, 0], 4, MarshalTypeConst);
        _test_marshal_int64array_size(&[100, 100, 100, 100], 4, MarshalTypeConst);
    }

    fn _test_marshal_int64array_size(va: &[i64],
                                     precision_bits: u8,
                                     min_size_expected: usize,
                                     max_size_expected: usize) {
        b, _, _ := marshalInt64Array(nil, va, precision_bits);
        if b.len() > max_size_expected {
            panic!("too big size for marshaled {} items with precision_bits {}: got {}; expecting {}",
                   va.len(), precision_bits, b.len(), max_size_expected)
        }
        if b.len() < min_size_expected {
            panic!("too small size for marshaled {} items with precision_bits {}: got {}; expecting {}",
                   va.len(), precision_bits, b.len(), min_size_expected)
        }
    }
}