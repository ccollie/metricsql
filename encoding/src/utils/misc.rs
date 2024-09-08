/// ensure_non_decreasing_sequence makes sure the first item in a is v_min, the last
/// item in a is v_max and all the items in a are non-decreasing.
///
/// If this isn't the case the a is fixed accordingly.
pub fn ensure_non_decreasing_sequence(a: &mut [i64], v_min: i64, v_max: i64) {
    if v_max < v_min {
        panic!("BUG: v_max cannot be smaller than v_min; got {v_max} vs {v_min}")
    }
    if a.is_empty() {
        return;
    }
    let max = v_max;
    let min = v_min;
    if a[0] != v_min {
        a[0] = min;
    }

    let mut v_prev = a[0];
    for value in a.iter_mut().skip(1) {
        if *value < v_prev {
            *value = v_prev;
        }
        v_prev = *value;
    }

    let mut i = a.len() - 1;
    if a[i] != max {
        a[i] = max;
        if i > 0 {
            i -= 1;
            while i > 0 && a[i] > max {
                a[i] = max;
                i -= 1;
            }
        }
    }
}

/// is_const returns true if a contains only equal values.
pub fn is_const(a: &[i64]) -> bool {
    if a.is_empty() {
        return false;
    }
    is_const_buf(a)
}

/// is_delta_const returns true if a contains counter with constant delta.
pub fn is_delta_const(a: &[i64]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let d1 = a[1] - a[0];
    let mut prev = a[1];
    for next in &a[2..] {
        if *next - prev != d1 {
            return false;
        }
        prev = *next;
    }
    true
}

pub fn is_const_buf<T: Copy + PartialEq>(data: &[T]) -> bool {
    if data.is_empty() {
        return true;
    }
    let comparator = [data[0]; 256];
    let mut cursor = data;
    while !cursor.is_empty() {
        let mut n = comparator.len();
        if n > cursor.len() {
            n = cursor.len()
        }
        let x = &cursor[0..n];
        if x != &comparator[0..x.len()] {
            return false;
        }
        cursor = &cursor[n..];
    }
    true
}

/// is_gauge returns true if a contains gauge values,
/// i.e. arbitrary changing values.
///
/// It is OK if a few gauges aren't detected (i.e. detected as counters),
/// since mis-detected counters as gauges leads to worse compression ratio.
pub fn is_gauge(a: &[i64]) -> bool {
    // Check all the items in a, since a part of items may lead
    // to incorrect gauge detection.

    if a.len() < 2 {
        return false;
    }

    let mut resets = 0;
    let mut v_prev = a[0];
    if v_prev < 0 {
        // Counter values cannot be negative.
        return true;
    }
    for v in &a[1..] {
        if *v < v_prev {
            if *v < 0 {
                // Counter values cannot be negative.
                return true;
            }
            if *v > (v_prev >> 3) {
                // Decreasing sequence detected.
                // This is a gauge.
                return true;
            }
            // Possible counter reset.
            resets += 1;
        }
        v_prev = *v;
    }
    if resets <= 2 {
        // Counter with a few resets.
        return false;
    }

    // Let it be a gauge if resets exceeds a.len()/8,
    // otherwise assume counter.
    resets > (a.len() >> 3)
}

#[cfg(test)]
mod tests {
    use super::{ensure_non_decreasing_sequence, is_const, is_delta_const, is_gauge};

    #[test]
    fn test_is_const() {
        let check = |a: &[i64], ok_expected: bool| {
            let ok = is_const(a);
            assert_eq!(
                ok, ok_expected,
                "unexpected is_const for a={:?}; got {}; want {}",
                a, ok, ok_expected
            );
        };
        check(&[], false);
        check(&[1], true);
        check(&[1, 2], false);
        check(&[1, 1], true);
        check(&[1, 1, 1], true);
        check(&[1, 1, 2], false);
    }

    #[test]
    fn test_is_delta_const() {
        let check = |a: &[i64], ok_expected: bool| {
            let ok = is_delta_const(a);
            assert_eq!(
                ok, ok_expected,
                "unexpected isDeltaConst for a={:?}; got {}; want {}",
                a, ok, ok_expected
            );
        };

        check(&[], false);
        check(&[1], false);
        check(&[1, 2], true);
        check(&[1, 2, 3], true);
        check(&[3, 2, 1], true);
        check(&[3, 2, 1, 0, -1, -2], true);
        check(&[3, 2, 1, 0, -1, -2, 2], false);
        check(&[1, 1], true);
        check(&[1, 2, 1], false);
        check(&[1, 2, 4], false);
    }

    #[test]
    fn test_is_gauge() {
        let check = |values: &[i64], ok_expected: bool| {
            let ok = is_gauge(values);
            assert_eq!(
                ok, ok_expected,
                "unexpected result for is_gauge({:?}); got {}; expecting {}",
                values, ok, ok_expected
            );
        };
        check(&[], false);
        check(&[0], false);
        check(&[1, 2], false);
        check(&[0, 1, 2, 3, 4, 5], false);
        check(&[0, -1, -2, -3, -4], true);
        check(&[0, 0, 0, 0, 0, 0, 0], false);
        check(&[1, 1, 1, 1, 1], false);
        check(&[1, 1, 2, 2, 2, 2], false);
        check(&[1, 17, 2, 3], false); // a single counter reset
        check(&[1, 5, 2, 3], true);
        check(&[1, 5, 2, 3, 2], true);
        check(&[-1, -5, -2, -3], true);
        check(&[-1, -5, -2, -3, -2], true);
        check(&[5, 6, 4, 3, 2], true);
        check(&[4, 5, 6, 5, 4, 3, 2], true);
        check(&[1064, 1132, 1083, 1062, 856, 747], true);
    }

    #[test]
    fn test_ensure_non_decreasing_sequence() {
        fn check(values: &[i64], min: i64, max: i64, expected: &[i64]) {
            let mut values = Vec::from(values);
            ensure_non_decreasing_sequence(&mut values, min, max);
            assert_eq!(
                values, expected,
                "unexpected a; got\n{:?}; expecting\n{:?}",
                values, expected
            );
        }

        check(&[], -1234, -34, &[]);
        check(&[123], -1234, -1234, &[-1234]);
        check(&[123], -1234, 345, &[345]);
        check(&[-23, -14], -23, -14, &[-23, -14]);
        check(&[-23, -14], -25, 0, &[-25, 0]);
        check(&[0, -1, 10, 5, 6, 7], 2, 8, &[2, 2, 8, 8, 8, 8]);
        check(&[0, -1, 10, 5, 6, 7], -2, 8, &[-2, -1, 8, 8, 8, 8]);
        check(&[0, -1, 10, 5, 6, 7], -2, 12, &[-2, -1, 10, 10, 10, 12]);
        check(&[1, 2, 1, 3, 4, 5], 1, 5, &[1, 2, 2, 3, 4, 5]);
    }
}
