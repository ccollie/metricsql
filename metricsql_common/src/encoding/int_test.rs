#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::{
        marshal_bytes, marshal_i64, marshal_u64, marshal_var_i64, unmarshal_bytes, unmarshal_i64,
        unmarshal_u64, unmarshal_var_i64,
    };

    fn check_prefix_suffix<T: Display>(v: T, buf: &[u8], buf_with_prefix: &[u8], prefix: &[u8]) {
        let new_prefix = &buf_with_prefix[0..prefix.len()];
        let suffix = &buf_with_prefix[prefix.len()..];
        assert_eq!(
            new_prefix, prefix,
            "unexpected prefix for v={}; got\n{:?}; expecting\n{:?}",
            v, new_prefix, prefix
        );
        assert_eq!(
            suffix, buf,
            "unexpected b for v={}; got\n{:?}; expecting\n{:?}",
            v, suffix, buf
        );
    }

    #[test]
    fn test_marshal_unmarshal_u64() {
        test_u64(0);
        test_u64(1);
        test_u64(u64::MAX - 1);
        test_u64((1 << 63) + 1);
        test_u64((1 << 63) - 1);
        test_u64(1 << 63);

        for i in 0..10000 {
            test_u64(i)
        }
    }

    fn test_u64(u: u64) {
        let mut b: Vec<u8> = vec![];
        marshal_u64(&mut b, u);
        assert_eq!(
            b.len(),
            8,
            "unexpected b length: {}; expecting {}",
            b.len(),
            8
        );
        let (u_new, _) = unmarshal_u64(&b).expect("Error unmarshalling u64");
        assert_eq!(
            u_new, u,
            "unexpected u_new from b={:?}; got {}; expecting {}",
            b, u_new, u
        );

        let prefix = [1, 2, 3];
        let mut b1: Vec<u8> = Vec::from(prefix);
        marshal_u64(&mut b1, u);

        check_prefix_suffix(u, &b, &b1, &prefix);
    }

    #[test]
    fn test_marshal_unmarshal_i64() {
        test_i64(0);
        test_i64(1);
        test_i64(-1);
        test_i64(-1 << 63);
        test_i64((-1 << 63) + 1);
        test_i64(i64::MAX - 1);

        for i in 0..10000 {
            test_i64(i);
            test_i64(-i)
        }
    }

    fn test_i64(v: i64) {
        let mut b: Vec<u8> = vec![];
        marshal_i64(&mut b, v);
        assert_eq!(
            b.len(),
            8,
            "unexpected b length: {}; expecting {}",
            b.len(),
            8
        );

        let (v_new, _) = unmarshal_i64(&b).expect("Error unmarshalling i64");
        assert_eq!(
            v_new, v,
            "unexpected v_new from b={:?}; got {}; expecting {}",
            b, v_new, v
        );
        let prefix = [1, 2, 3];

        let mut b1: Vec<u8> = Vec::from(prefix);
        marshal_i64(&mut b1, v);

        check_prefix_suffix(v, &b, &b1, &prefix);
    }

    #[test]
    fn test_marshal_unmarshal_var_i64() {
        test_varint_i64(0);
        test_varint_i64(1);
        test_varint_i64(-1);
        test_varint_i64(-1 << 63);
        test_varint_i64((-1 << 63) + 1);
        test_varint_i64(i64::MAX - 1);

        for i in 0..1_0000 {
            test_varint_i64(i);
            test_varint_i64(-i);
            test_varint_i64(i << 8);
            test_varint_i64(-i << 8);
            test_varint_i64(i << 16);
            test_varint_i64(-i << 16);
            test_varint_i64(i << 23);
            test_varint_i64(-i << 23);
            test_varint_i64(i << 33);
            test_varint_i64(-i << 33);
            test_varint_i64(i << 43);
            test_varint_i64(-i << 43);
            test_varint_i64(i << 53);
            test_varint_i64(-i << 53)
        }
    }

    fn test_varint_i64(v: i64) {
        let mut b: Vec<u8> = vec![];
        marshal_var_i64(&mut b, v);
        let (v_new, tail) = unmarshal_var_i64(&b).unwrap_or_else(|_| {
            panic!("unexpected error when unmarshalling v={} from b={:?}", v, b)
        });
        assert_eq!(
            v_new, v,
            "unexpected v_new from b={:?}; got {}; expecting {}",
            b, v_new, v
        );
        assert_eq!(
            tail.len(),
            0,
            "unexpected data left after unmarshalling v={} from b={:?}: {:?}",
            v,
            b,
            tail
        );

        let prefix = [1, 2, 3];
        let mut b1: Vec<u8> = Vec::from(prefix);
        marshal_var_i64(&mut b1, v);

        check_prefix_suffix(v, &b, &b1, &prefix);
    }

    #[test]
    fn test_marshal_unmarshal_bytes() {
        test_bytes("");
        test_bytes("x");
        test_bytes("xy");

        let mut bb: Vec<u8> = Vec::with_capacity(600);
        for i in 0..100 {
            bb.extend_from_slice(format!(" {} ", i).as_bytes());
            let s = String::from_utf8(bb.clone()).expect("error in buf => utf-8 conversion ");
            test_bytes(&s)
        }
    }

    fn test_bytes(s: &str) {
        let mut b: Vec<u8> = vec![];
        marshal_bytes(&mut b, s.as_bytes());
        let (b_new, tail) = unmarshal_bytes(&b).unwrap_or_else(|_| {
            panic!("unexpected error when unmarshalling s={} from b={:?}", s, b)
        });

        assert_eq!(
            tail.len(),
            0,
            "unexpected data left after unmarshalling s={} from b={:?}: {:?}",
            s,
            b,
            tail
        );

        assert_eq!(
            b_new,
            s.as_bytes(),
            "unexpected sNew from b={:?}; got {:?}; expecting {}",
            b,
            b_new,
            s
        );

        let prefix = b"abcde".as_slice();
        let mut b1: Vec<u8> = Vec::from(prefix);
        marshal_bytes(&mut b1, s.as_bytes());

        check_prefix_suffix(s, &b, &b1, prefix);
    }
}
