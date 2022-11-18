use rand::{Rng, thread_rng};
use rand_distr::StandardNormal;
use crate::error::Error;
use crate::{marshal_int64_array, MarshalType, unmarshal_int64_array};

pub fn get_rand_normal() -> f64 {
    thread_rng().sample::<f64,_>(StandardNormal)
}

pub fn check_precision_bits(a: &[i64], b: &[i64], precision_bits: u8) -> Result<(), Error> {
    if a.len() != b.len() {
        let msg = format!("different-sized arrays: {} vs {}", a.len(), b.len());
        return Err(Error::new(msg));
    }
    let mut i = 0;
    for (i, av) in a.iter().enumerate() {
        let mut av: i64 = *av;
        let mut bv = b[i];
        if av < bv {
            let (av, bv) = (bv, av);
        }
        let eps = av - bv;
        if eps == 0 {
            continue
        }
        if av < 0 {
            av = -av
        }
        let mut pbe = 1_u8;
        while eps < av {
            av >>= 1;
            pbe += 1;
        }
        if pbe < precision_bits {
            let msg = format!("too low precision_bits for\na={:?}\nb={:?}\ngot {}; expecting {}; compared values: {} vs {}, eps={}",
                              a, b, pbe, precision_bits, a[i], b[i], eps);
            return Err(Error::new(msg.as_str()));
        }
    }

    Ok(())
}

pub fn ensure_marshal_unmarshal_int64_array(va: &[i64], precision_bits: u8, mt_expected: MarshalType) {
    use MarshalType::*;

    let mut b: Vec<u8> = vec![];
    let (mt, first_value) = marshal_int64_array(&mut b, va, precision_bits)
        .expect("marshal_int64_array");
    assert_eq!(mt, mt_expected,
               "unexpected MarshalType for va={:?}, precision_bits={}: got {}; expecting {}",
               va, precision_bits, mt, mt_expected);

    let mut va_new: Vec<i64> = vec![];
    unmarshal_int64_array(&mut va_new, &b, mt, first_value, va.len())
        .expect("unmarshal_int64_array");

    match mt {
        Lz4NearestDelta | Lz4NearestDelta2 |
        NearestDelta | NearestDelta2 => {
            match check_precision_bits(va, &va_new, precision_bits) {
                Err(err) => panic!("too low precision for va_new: {:?}", err),
                _ => {}
            }
        },
        _ => {
            assert_eq!(va, va_new,
                       "unexpected va_new for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
                       va, precision_bits, va_new, va)
        }
    }

    let b_prefix = [1, 2, 3];
    let mut b_new: Vec<u8> = Vec::from(b_prefix);
    let (mt_new, first_value_new) = marshal_int64_array(&mut b_new, va, precision_bits)
        .expect("marshal_int64_array");
    assert_eq!(first_value_new, first_value,
               "unexpected first_value for prefixed va={:?}, precision_bits={}; got {}; want {}",
               va, precision_bits, first_value_new, first_value);

    let new_prefix = &b_new[0..b_prefix.len()];
    assert_eq!(new_prefix, b_prefix, "unexpected prefix for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
               va, precision_bits, new_prefix, b_prefix);

    let suffix = &b_new[b_prefix.len()..];
    assert_eq!(suffix, &b,
               "unexpected b for prefixed va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
               va, precision_bits, suffix, b);

    assert_eq!(mt_new, mt, "unexpected mt for prefixed va={:?}, precision_bits={}; got {}; expecting {}",
               va, precision_bits, mt_new, mt);

    let va_prefix = [4, 5, 6, 8];
    let mut vaNew: Vec<i64> = Vec::from(va_prefix);
    unmarshal_int64_array(&mut vaNew, &b, mt, first_value, va.len())
        .expect("unmarshal_int64_array");

    let prefix = &vaNew[0 .. va_prefix.len()];
    assert_eq!(prefix, &va_prefix,
               "unexpected prefix for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
               va, precision_bits, prefix, va_prefix);

    match mt {
        Lz4NearestDelta |
        Lz4NearestDelta2 |
        NearestDelta |
        NearestDelta2 => {
            let suffix = &vaNew[va_prefix.len() .. ];
            match check_precision_bits(suffix, &va, precision_bits) {
                Err(err) => panic!("too low precision for timestamps: {:?}", err),
                _ => {}
            }
        },
        _ => {
            let suffix = &vaNew[prefix.len() .. ];
            assert_eq!(suffix, va,
                       "unexpected prefixed va_new for va={:?}, precision_bits={}; got\n{:?}; expecting\n{:?}",
                       va, precision_bits, suffix, va)
        }
    }
}

pub fn ensure_marshal_int64_array_size(va: &[i64],
                                 precision_bits: u8,
                                 min_size_expected: usize,
                                 max_size_expected: usize) {
    let mut b: Vec<u8> = vec![];
    marshal_int64_array(&mut b, va, precision_bits).expect("marshal_int64_array");
    if b.len() > max_size_expected {
        panic!("too big size for marshaled {} items with precision_bits {}: got {}; expecting {}",
               va.len(), precision_bits, b.len(), max_size_expected)
    }
    if b.len() < min_size_expected {
        panic!("too small size for marshaled {} items with precision_bits {}: got {}; expecting {}",
               va.len(), precision_bits, b.len(), min_size_expected)
    }
}