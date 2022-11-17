#[cfg(test)]
mod tests {

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
    
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 1, -5e2, "00000000");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 2, -5e2, "0000ff0300");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 3, -5e2, "00ff01ff01ff01");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 4, -5e2, "7fff017fff01");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 5, -5e2, "bf01bf01bf01bf01");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 6, -5e2, "bf01bf01bf01bf01");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 7, -5e2, "bf01cf01bf01af01");
    check_marshal_unmarshal_i64(&[-5e2, -6e2, -7e2, -8e2, -8.9e2], 8, -5e2, "c701c701c701af01");
}

fn check_marshal_unmarshal_i64(va: &[i64], precision_bits: u8, first_value_expected: i64, b_expected: &str) {
    let mut b: Vec<8> = vec![];
    let first_value = marshal_int64_nearest_delta(&mut b, va, precision_bits);
    assert_eq!(first_value, first_value_expected, "unexpected first_value for va={}, precision_bits={}; got {}; want {}",
               va, precision_bits, first_value, first_value_expected);

    if fmt.Sprintf("%x", b) != b_expected {
        t.Fatalf("invalid marshaled data for va={}, precision_bits={}; got\n%x; expecting\n%s", va, precision_bits, b, b_expected)
    }

    let prefix = b"foobar".as_slice();
    b, first_value = marshal_int64_nearest_delta(prefix, va, precision_bits);
    assert_eq!(first_value, first_value_expected,
               "unexpected first_value for va={}, precision_bits={}; got {}; want {}", va, precision_bits, first_value, first_value_expected);

    let new_prefix = &b[0 .. prefix.len()];
    assert_eq!(new_prefix, prefix,
        "invalid prefix for va={}, precision_bits={}; got\n{:?}; expecting\n{:?}",
        va, precision_bits, new_prefix, prefix);

    let suffix = &b[prefix.len()..];
    assert_eq!(suffix, b_expected.as_bytes(),
        "invalid marshaled prefixed data for va={}, precision_bits={}; got\n{:?}; expecting\n{}",
        va, precision_bits, suffix, b_expected)
}

#[test]
fn test_marshal_unmarshal_int64nearest_delta() {
    check_int64_nearest_delta(&[0], 4);
    check_int64_nearest_delta(&[0, 0], 4);
    check_int64_nearest_delta(&[1, -3], 4);
    check_int64_nearest_delta(&[255, 255], 4);
    check_int64_nearest_delta(&[0, 1, 2, 3, 4, 5], 4);
    check_int64_nearest_delta(&[5, 4, 3, 2, 1, 0], 4);
    check_int64_nearest_delta(&[-5e12, -6e12, -7e12, -8e12, -8.9e12], 1);
    check_int64_nearest_delta(&[-5e12, -6e12, -7e12, -8e12, -8.9e12], 2);
    check_int64_nearest_delta(&[-5e12, -6e12, -7e12, -8e12, -8.9e12], 3);
    check_int64_nearest_delta(&[-5e12, -5.6e12, -7e12, -8e12, -8.9e12], 4);

    let mut rng = rand::thread_rng();

    // Verify constant encoding.
    let va: Vec<i64> = Vec::with_capacity(1024);
    for i in 0 .. 1024 {
        va.push(9876543210123)
    }
    check_int64_nearest_delta(va, 4);
    check_int64_nearest_delta(va, 63);

    // Verify encoding for monotonically incremented va.
    let v = int64(-35);
    let va: Vec<i64> = Vec::with_capacity(1024);
    for i in 0 .. 1024 {
        v += 8;
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);
    check_int64_nearest_delta(va, 63);

    // Verify encoding for monotonically decremented va.
    let mut v = 793;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for _ in 0 .. 1024 {
        v -= 16;
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);
    check_int64_nearest_delta(va, 63);

    // Verify encoding for quadratically incremented va.
    v = -1234567;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for i in 0 .. 1024 {
        v += 32 + i as i64;
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);

    // Verify encoding for decremented va with norm-float noise.
    v = 787933;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for _ in 0 .. 1024 {
        v -= 25 + int64(rand.NormFloat64()*2);
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);

    // Verify encoding for incremented va with random noise.
    v = 943854;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for _ in 0 .. 1024 {
        v += 30 + rand.Int63n(5);
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);

    // Verify encoding for constant va with norm-float noise.
    v = -12345;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for _ in 0 .. 1024 {
        v += int64(rand.NormFloat64() * 10);
        va.push(v);
    }
    check_int64_nearest_delta(va, 4);

    // Verify encoding for constant va with random noise.
    v = -12345;
    let va: Vec<i64> = Vec::with_capacity(1024);
    for _ in 0 .. 1024 {
        v += rand.Int63n(15) - 1;
        va.push(v);
    }
    check_int64_nearest_delta(va, 4)
}

fn check_int64_nearest_delta(va: &[i64], precision_bits: uint8) {
    let b: Vec<u8> = vec![];
    let first_value = marshal_int64_nearest_delta(&mut dst, va, precision_bits);
    let vaNew = unmarshal_int64_nearest_delta(&dst, b, first_value, va.len());
    if err != nil {
        t.Fatalf("cannot unmarshal data for va={}, precision_bits={} from b=%x: %s", va, precision_bits, b, err)
    }

    match check_precision_bits(vaNew, va, precision_bits) {
        Err(err) => {
            panic!("too small precision_bits for va={:?}, precision_bits={}: {:?}, vaNew=\n{}", va,
                   precision_bits, err, vaNew)
        },
        Ok(_) => {}
    }

    let vaPrefix = [1, 2, 3, 4];
    let buf: Vec<u8> = vec![];
    vaNew, err = unmarshal_int64_nearest_delta(vaPrefix, b, first_value, va.len());
    if err != nil {
        panic!("cannot unmarshal prefixed data for va={:?}, precision_bits={} from b={:?}: {}",
                 va, precision_bits, b, err)
    }

    let prefix = &vaNew[0 .. vaPrefix.len()];
    assert_eq!(prefix, vaPrefix,
        "unexpected prefix for va={:?}, precision_bits={}: got\n{:?}; expecting\n{:?}",
        va, precision_bits, prefix, vaPrefix);

    let suffix = &vaNew[vaPrefix.len()..];
    match check_precision_bits(suffix, va, precision_bits) {
        Err(e) => {
            panic!("too small precision_bits for va={:?}, precision_bits={}: {:?}, vaNew=\n{}", va,
                   precision_bits, err, suffix)
        },
        _ => {}
    }

    match check_precision_bits(vaNew, va, precision_bits) {
        Err(err) => {
            panic!("too small precision_bits for va={:?}, precision_bits={}: {:?}, vaNew=\n{}", va,
                   precision_bits, err, vaNew)
        },
        Ok(_) => {}
    }
}
}