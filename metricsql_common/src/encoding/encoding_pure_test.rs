#[cfg(test)]
mod tests {
    use crate::tests::utils::*;
    use crate::MarshalType::{NearestDelta, NearestDelta2};

    #[test]
    fn test_marshal_unmarshal_int64array() {
        let mut va: Vec<i64> = Vec::with_capacity(8 * 1024);
        let mut v: i64 = 0;

        // Verify nearest delta encoding.
        for _ in 0..8 * 1024 {
            v += (get_rand_normal() * 1e6) as i64;
            va.push(v);
        }

        for precision_bits in 1..17 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta)
        }

        for precision_bits in 23..65 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta)
        }

        va.clear();
        // Verify nearest delta2 encoding.
        v = 0;

        for _ in 0..8 * 1024 {
            v += (30e6 + get_rand_normal() * 1e6) as i64;
            va.push(v);
        }

        for precision_bits in 1..15 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta2)
        }

        for precision_bits in 24..65 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta2)
        }

        // Verify nearest delta encoding.
        va.clear();
        v = 1000;
        for _ in 0..6 {
            v += (get_rand_normal() * 100_f64) as i64;
            va.push(v);
        }

        for precision_bits in 1..65 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta)
        }

        // Verify nearest delta2 encoding.
        va.clear();
        v = 0;
        for _ in 0..6 {
            v += 3000 + (get_rand_normal() * 100_f64) as i64;
            va.push(v);
        }

        for precision_bits in 5..65 {
            ensure_marshal_unmarshal_int64_array(&va, precision_bits, NearestDelta2)
        }
    }

    #[test]
    fn test_marshal_int64_array_size() {
        let mut va: Vec<i64> = Vec::with_capacity(8 * 1024);

        let mut v = (get_rand_normal() * 1e9) as i64;
        for _ in 0..8 * 1024 {
            va.push(v);
            v += (30e3 + get_rand_normal() * 1e3) as i64;
        }

        let va = &va;
        ensure_marshal_int64_array_size(va, 1, 500, 1700);
        ensure_marshal_int64_array_size(va, 2, 600, 1800);
        ensure_marshal_int64_array_size(va, 3, 900, 2100);
        ensure_marshal_int64_array_size(va, 4, 1300, 2200);
        ensure_marshal_int64_array_size(va, 5, 2000, 3300);
        ensure_marshal_int64_array_size(va, 6, 3000, 5000);
        ensure_marshal_int64_array_size(va, 7, 4000, 6500);
        ensure_marshal_int64_array_size(va, 8, 6000, 8000);
        ensure_marshal_int64_array_size(va, 9, 7000, 8800);
        ensure_marshal_int64_array_size(va, 10, 8000, 17000)
    }
}
