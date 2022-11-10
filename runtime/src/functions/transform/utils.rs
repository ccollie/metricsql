use std::cmp::Ordering;
use metricsql::utils::parse_float;
use crate::{RuntimeError, RuntimeResult};

pub(super) fn cmp_alpha_numeric(a: &str, b: &str) -> RuntimeResult<Ordering> {
    let (mut a, mut b) = (a, b);
    loop {
        if b.is_empty() {
            return Ok(Ordering::Greater);
        }
        if a.is_empty() {
            return Ok(Ordering::Less);
        }
        let a_prefix = get_num_prefix(a);
        let b_prefix = get_num_prefix(b);
        let a_len = a_prefix.len();
        let b_len = b_prefix.len();
        
        a = &a[a_len .. ];
        b = &b[b_len .. ];
        if a_len > 0 || b_len > 0 {
            if a_len == 0 {
                return Ok(Ordering::Greater);
            }
            if b_len == 0 {
                return Ok(Ordering::Less);
            }
            let a_num = must_parse_num(a_prefix)?;
            let b_num = must_parse_num(b_prefix)?;
            if a_num != b_num {
                return Ok(a_num.total_cmp(&b_num));
            }
        }
        let a_non_numeric = get_non_num_prefix(a);
        let b_non_numeric_prefix = get_non_num_prefix(b);
        a = &a[a_non_numeric.len() .. ];
        b = &b[b_non_numeric_prefix.len() .. ];
        if a_non_numeric != b_non_numeric_prefix {
            return Ok(a_non_numeric.cmp(b_non_numeric_prefix));
        }
    }
}

pub(super) fn get_num_prefix(s: &str) -> &str {
    let mut i = 0;

    if !s.is_empty() {
        let ch = s.chars().next().unwrap();
        if ch == '-' || ch == '+' {
            i += 1;
        }
    }

    let mut has_num = false;
    let mut has_dot = false;
    for (j, ch) in s.chars().enumerate() {
        if !is_decimal_char(ch) {
            if !has_dot && ch == '.' {
                has_dot = true;
                i = j;
                continue
            }
            if !has_num {
                return ""
            }
            return &s[0..i];
        }
        has_num = true;
        i += 1;
    }

    if !has_num {
        return ""
    }
    s
}

fn get_non_num_prefix(s: &str) -> &str {
    for (i, ch) in s.chars().enumerate() {
        if is_decimal_char(ch) {
            return &s[0..i];
        }
    }
    return s
}

fn is_decimal_char(ch: char) -> bool {
    return ch >= '0' && ch <= '9'
}

fn must_parse_num(s: &str) -> RuntimeResult<f64> {
    match parse_float(s) {
        Err(err) => {
            Err(RuntimeError::from(format!("BUG: unexpected error when parsing the number {}: {:?}", s, err)))
        },
        Ok(v) => Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering::{Equal, Greater, Less};
    use metricsql::utils::parse_float;
    use crate::functions::transform::utils::{cmp_alpha_numeric, get_num_prefix};

    fn test_prefix(s: &str, expected_prefix: &str) {
        let prefix = get_num_prefix(s);
        assert_eq!(prefix, prefix_expected, "unexpected get_num_prefix({}): got {}; want {}", s, prefix, prefix_expected);
        if prefix.len() > 0 {
            parse_float(prefix).expect(format!("unable to parse {} as float", prefix).as_str())
        }
    }
    
    #[test]
    fn test_get_num_prefix() {
        test_prefix("", "");
        test_prefix("foo", "");
        test_prefix("-", "");
        test_prefix(".", "");
        test_prefix("-.", "");
        test_prefix("+..", "");
        test_prefix("1", "1");
        test_prefix("12", "12");
        test_prefix("1foo", "1");
        test_prefix("-123", "-123");
        test_prefix("-123bar", "-123");
        test_prefix("+123", "+123");
        test_prefix("+123.", "+123.");
        test_prefix("+123..", "+123.");
        test_prefix("+123.-", "+123.");
        test_prefix("12.34..", "12.34");
        test_prefix("-12.34..", "-12.34");
        test_prefix("-12.-34..", "-12.")
    }

    #[test]
    fn test_cmp_alpha_numeric() {
        use std::cmp::Ordering;

        let f = |a: &str, b: &str, want: Ordering| {
            let got = cmp_alpha_numeric(a, b).unwrap();
            assert_eq!(got, want, "unexpected cmp_alpha_numeric({}, {}): got {}; want {}", a, b, got, want);
        };

        // empty strings
        f("", "", Equal);
        f("", "321", Less);
        f("321", "", Greater);
        f("", "abc", Less);
        f("abc", "", Greater);
        f("foo", "123", Greater);
        f("123", "foo", Less);
        // same length numbers
        f("123", "321", Less);
        f("321", "123", Greater);
        f("123", "123", Equal);
        // same length strings
        f("a", "b", Less);
        f("b", "a", Greater);
        f("a", "a", Equal);
        // identical string prefix
        f("foo123", "foo", Greater);
        f("foo", "foo123", Less);
        f("foo", "foo", Equal);
        // identical num prefix
        f("123foo", "123bar", Greater);
        f("123bar", "123foo", Less);
        f("123bar", "123bar", Equal);
        // numbers with special chars
        f("1:0:0", "1:0:2", Less);
        // numbers with special chars and different number rank
        f("1:0:15", "1:0:2", Greater);
        // multiple zeroes"
        f("0", "00", Equal);
        // only chars
        f("aa", "ab", Less);
        // strings with different lengths
        f("ab", "abc", Less);
        // multiple zeroes after equal char
        f("a0001", "a0000001", Greater);
        // short first string with numbers and highest rank
        f("a10", "abcdefgh2", Less);
        // less as second string
        f("a1b", "a01b", Greater);
        // equal strings by length with different number rank
        f("a001b01", "a01b001", Greater);
        // different numbers rank
        f("a01b001", "a001b01", Greater);
        // different numbers rank
        f("a01b001", "a001b01", Greater);
        // highest char and number
        f("a1", "a1x", Less);
        // highest number reverse chars
        f("1b", "1ax", Greater);
        // numbers with leading zero
        f("082", "83", Less);
        // numbers with leading zero and chars
        f("083a", "9a", Greater);
        f("083a", "94a", Less);
        // negative number
        f("-123", "123", Less);
        f("-123", "+123", Less);
        f("-123", "-123", Equal);
        f("123", "-123", Greater);
        // fractional number
        f("12.9", "12.56", Greater);
        f("12.56", "12.9", Less);
        f("12.9", "12.9", Equal);
    }
}