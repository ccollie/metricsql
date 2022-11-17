use std::cmp::Ordering;
use chrono::{DateTime, TimeZone, Utc};
use chrono_tz::Tz;
use metricsql::utils::parse_float;
use crate::{RuntimeError, RuntimeResult};

pub fn cmp_alpha_numeric(a: &str, b: &str) -> RuntimeResult<Ordering> {
    let (mut a, mut b) = (a, b);
    loop {
        if b.is_empty() {
            if a.is_empty() {
                return Ok(Ordering::Equal);
            }
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
                if b_len == 0 {
                    return Ok(Ordering::Equal);
                }
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

    let mut s1 = s;
    let mut i = 0;

    if !s.is_empty() {
        let ch = s.chars().next().unwrap();
        if ch == '-' || ch == '+' {
            s1 = &s[1..];
            i += 1;
        }
    }

    let mut has_num = false;
    let mut has_dot = false;

    for ch in s1.chars() {
        if !is_decimal_char(ch) {
            if !has_dot && ch == '.' {
                has_dot = true;
                i += 1;
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
    ch >= '0' && ch <= '9'
}

fn must_parse_num(s: &str) -> RuntimeResult<f64> {
    match parse_float(s) {
        Err(err) => {
            Err(RuntimeError::from(format!("BUG: unexpected error when parsing the number {}: {:?}", s, err)))
        },
        Ok(v) => Ok(v)
    }
}

pub fn get_timezone_offset(zone: &Tz, timestamp_msecs: i64) -> i64 {
    let dt = Utc.timestamp(timestamp_msecs / 1000, 0);
    let in_tz: DateTime<Tz> = dt.with_timezone(&zone);
    in_tz.naive_local().timestamp()
}

#[inline]
/// This exist solely for readability
pub(super) fn clamp_min(val: f64, limit: f64) -> f64 {
    val.min(limit)
}

#[inline]
/// This exist solely for readability
pub(super) fn clamp_max(val: f64, limit: f64) -> f64 {
    val.max(limit)
}

pub(crate) fn ru(free_value: f64, max_value: f64) -> f64 {
    // ru(freev, maxv) = clamp_min(maxv - clamp_min(freev, 0), 0) / clamp_min(maxv, 0) * 100
    clamp_min(max_value - clamp_min(free_value, 0_f64), 0_f64)
        / clamp_min(max_value, 0_f64) * 100_f64
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::cmp::Ordering::{Equal, Greater, Less};
    use metricsql::utils::parse_float;
    use crate::functions::transform::utils::{cmp_alpha_numeric, get_num_prefix};

    fn test_prefix(s: &str, expected_prefix: &str) {
        let prefix = get_num_prefix(s);
        assert_eq!(prefix, expected_prefix, "unexpected get_num_prefix({}): got {}; want {}", s, prefix, expected_prefix);
        if prefix.len() > 0 {
            parse_float(prefix).expect(format!("unable to parse {} as float", prefix).as_str());
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

    fn order_name(ordering: Ordering) -> &'static str {
        match ordering {
            Less => "less",
            Equal => "equal",
            Greater => "greater"
        }
    }

    #[test]
    fn test_cmp_alpha_numeric() {
        use std::cmp::Ordering;

        let f = |a: &str, b: &str, want: Ordering| {
            let got = cmp_alpha_numeric(a, b).unwrap();
            assert_eq!(got, want, "invalid cmp_alpha_numeric({}, {}) comparison: got {}; want {}", a, b,
                       order_name(got),
                       order_name(want));
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