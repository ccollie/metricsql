#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::{marshal_bytes, marshal_u64, unmarshal_bytes, unmarshal_u64};

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
