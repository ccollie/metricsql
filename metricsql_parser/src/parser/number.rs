use crate::parser::ParseError::InvalidNumber;
use crate::parser::{ParseError, ParseResult};

fn from_str_radix(str: &str, radix: u32) -> Result<f64, ParseError> {
    match u64::from_str_radix(str, radix) {
        Ok(value) => Ok(value as f64),
        Err(_) => Err(InvalidNumber(str.to_string())),
    }
}

fn parse_basic(str: &str) -> ParseResult<f64> {
    let mut multiplier = 1;
    let mut str = str;
    if let Some((ending, mult)) = get_number_suffix(str) {
        str = &str[0..str.len() - ending.len()];
        multiplier = *mult;
    }
    match str.parse::<f64>() {
        Ok(value) => Ok(if multiplier > 1 {
            value * multiplier as f64
        } else {
            value
        }),
        Err(_) => Err(InvalidNumber(str.to_string())),
    }
}

/// Note. This should only called on strings produced by the lexer. To be more
/// specific, this assumes strings are ascii
pub fn parse_positive_number(str: &str) -> ParseResult<f64> {
    if str.is_empty() {
        return Err(InvalidNumber(str.to_string()));
    }
    let ch = str.chars().next().unwrap();
    match ch {
        'i' | 'I' => {
            // perf: avoid allocation by exhaustive check of casing permutations instead of
            // first converting case
            if matches!(
                str,
                "inf" | "Inf" | "iNf" | "inF" | "INf" | "InF" | "iNF" | "INF"
            ) {
                return Ok(f64::INFINITY);
            }
        }
        'n' | 'N' => {
            // perf: avoid allocation by exhaustive check of casing permutations instead of
            // first converting case
            if matches!(
                str,
                "nan" | "Nan" | "nAn" | "naN" | "NAn" | "NaN" | "nAN" | "NAN"
            ) {
                return Ok(f64::NAN);
            }
        }
        '0' => {
            if str.len() == 1 {
                return Ok(0_f64);
            }
            let rest = &str[1..];
            let ch = rest.chars().next().unwrap();
            return match ch {
                '.' => parse_basic(str),
                'b' | 'B' => from_str_radix(&rest[1..], 2),
                'o' | 'O' => from_str_radix(&rest[1..], 8),
                'x' | 'X' => from_str_radix(&rest[1..], 16),
                _ => {
                    if ch.is_numeric() {
                        // try and match Go style octal
                        return from_str_radix(rest, 8);
                    }
                    // punt to std metricsql_common num parsing
                    parse_basic(str)
                }
            };
        }
        _ => return parse_basic(str),
    }

    Err(InvalidNumber(str.to_string()))
}

pub fn parse_number(str: &str) -> ParseResult<f64> {
    let mut str = str;
    let ch = str.chars().next().unwrap();
    let mut is_negative = false;

    if ch == '+' {
        str = &str[1..];
    } else {
        is_negative = if ch == '-' {
            str = &str[1..];
            true
        } else {
            false
        };
    }

    parse_positive_number(str).map(|value| if is_negative { -1.0 * value } else { value })
}

type SuffixValue = (&'static str, usize);
const SUFFIXES: [SuffixValue; 16] = [
    ("kib", 1024),
    ("ki", 1024),
    ("kb", 1000),
    ("k", 1000),
    ("mib", 1024 * 1024),
    ("mi", 1024 * 1024),
    ("mb", 1000 * 1000),
    ("m", 1000 * 1000),
    ("gib", 1024 * 1024 * 1024),
    ("gi", 1024 * 1024 * 1024),
    ("gb", 1000 * 1000 * 1000),
    ("g", 1000 * 1000 * 1000),
    ("tib", 1024 * 1024 * 1024 * 1024),
    ("ti", 1024 * 1024 * 1024 * 1024),
    ("tb", 1000 * 1000 * 1000 * 1000),
    ("t", 1000 * 1000 * 1000 * 1000),
];

pub fn get_number_suffix(s: &str) -> Option<&'static SuffixValue> {
    if s.is_empty() {
        return None;
    }
    let last_ch = s.chars().last().unwrap();
    if last_ch.is_alphabetic() {
        // todo: avoid converrsion here
        let lower = s.to_ascii_lowercase();
        SUFFIXES.iter().find(|x| lower.ends_with(x.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::number::parse_positive_number;

    fn expect_failure(s: &str) {
        match parse_positive_number(s) {
            Err(..) => {}
            Ok(ns) => {
                panic!(
                    "expecting error in parse_positive_number({}); got result {}",
                    s, ns
                )
            }
        }
    }

    #[test]
    fn test_parse_number_with_unit() {
        fn f(s: &str, expected: f64) {
            let v = parse_positive_number(s).unwrap();
            if v.is_nan() {
                assert!(
                    expected.is_nan(),
                    "unexpected value returned from parse_number_with_unit({}); got {}; want {}",
                    s,
                    v,
                    expected
                )
            } else {
                assert_eq!(
                    v, expected,
                    "unexpected value returned from parse_number_with_unit({}); got {}; want {}",
                    s, v, expected
                )
            }
        }

        f("2k", (2 * 1000) as f64);
        f("2.3Kb", 2.3 * 1000_f64);
        f("3ki", (3 * 1024) as f64);
        f("4.5Kib", 4.5 * 1024_f64);
        f("2m", (2 * 1000 * 1000) as f64);
        f("2.3Mb", 2.3 * 1000_f64 * 1000_f64);
        f("3Mi", (3_i64 * 1024 * 1024) as f64);
        f("4.5mib", 4.5 * (1024 * 1024) as f64);
        f("2G", (2_i64 * 1000 * 1000 * 1000) as f64);
        f("2.3gB", 2.3 * (1000 * 1000 * 1000) as f64);
        f("3gI", (3_i64 * 1024 * 1024 * 1024) as f64);
        f("4.5GiB", 4.5 * (1024_i64 * 1024 * 1024) as f64);
        f("2T", (2_i64 * 1000 * 1000 * 1000 * 1000) as f64);
        f("2.3tb", 2.3 * (1000_i64 * 1000 * 1000 * 1000) as f64);
        f("3tI", (3 * 1024_i64 * 1024 * 1024 * 1024) as f64);
        f("4.5TIB", 4.5 * (1024_i64 * 1024 * 1024 * 1024) as f64)
    }

    #[test]
    fn test_parse_positive_number_success() {
        fn f(s: &str, expected: f64) {
            match parse_positive_number(s) {
                Err(err) => {
                    panic!(
                        "unexpected error in parse_positive_number({}): {:?}",
                        s, err
                    )
                }
                Ok(v) => {
                    if v.is_nan() {
                        if !expected.is_nan() {
                            panic!("unexpected value returned from parse_positive_number({}); got {}; want {}", s, v, expected)
                        }
                    } else if v != expected {
                        panic!("unexpected value returned from parse_positive_number({}); got {}; want {}", s, v, expected)
                    }
                }
            }
        }
        f("123", 123.0);
        f("1.23", 1.23);
        f("12e5", 12e5);
        f("1.3E-3", 1.3e-3);
        f("234.", 234.0);
        f("Inf", f64::INFINITY);
        f("NaN", f64::NAN);
        f("0xfe", 0xfe as f64);
        f("0b0110", 0b0110 as f64);
        f("0O765", 0o765 as f64);
        f("0765", 0o765 as f64);
        f("2k", (2 * 1000) as f64);
        f("2.3Kb", 2.3 * 1000_f64);
        f("3ki", 3.0 * 1024_f64);
        f("4.5Kib", 4.5 * 1024_f64);
        f("2m", 2.0 * (1000 * 1000) as f64);
        f("2.3Mb", 2.3 * (1000 * 1000) as f64);
        f("3Mi", 3.0 * 1024.0 * 1024.0);
        f("4.5mib", 4.5 * 1024.0 * 1024.0);
        f("2G", 2.0 * 1000.0 * 1000.0 * 1000.0);
        f("2.3gB", 2.3 * 1000.0 * 1000.0 * 1000.0);
        f("3gI", 3.0 * 1024.0 * 1024.0 * 1024.0);
        f("4.5GiB", 4.5 * 1024.0 * 1024.0 * 1024.0);
        f("2T", 2.0 * 1000.0 * 1000.0 * 1000.0 * 1000.0);
        f("2.3tb", 2.3 * 1000.0 * 1000.0 * 1000.0 * 1000.0);
        f("3tI", 3.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0);
        f("4.5TIB", 4.5 * 1024.0 * 1024.0 * 1024.0 * 1024.0)
    }

    #[test]
    fn test_parse_positive_number_failure() {
        fn f(s: &str) {
            expect_failure(s)
        }
        f("");
        f("0xqwert");
        f("foobar");
        f("234.foobar");
        f("123e");
        f("1233Ebc");
        f("12.34E+abc");
        f("12.34e-");
        f("12.weKB")
    }
}
