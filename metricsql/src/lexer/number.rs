use crate::parser::{ParseError, ParseResult};

#[inline]
fn parse_with_radix(str: &str, radix: u32, is_negative: bool) -> Result<f64, ParseError> {
    match u64::from_str_radix(str, radix) {
        Ok(n) => {
            let value = if is_negative {
                n as f64 * -1.0
            } else {
                n as f64
            };
            Ok(value)
        },
        Err(_) => Err(ParseError::InvalidNumber(str.to_string())),
    }
}

// todo: rename
pub fn parse_float(str: &str) -> Result<f64, ParseError> {
    let binding = str.to_ascii_lowercase();
    let mut str = binding.as_str();
    let ch = str.chars().next().unwrap();
    let is_negative = if ch == '-' {
        str = &str[1..];
        true
    } else {
        false
    };

    if str.len() > 2 {
        let prefix = &str[0..2];
        match prefix {
            "0b" => return parse_with_radix(&str[2..], 2, is_negative),
            "0o" => return parse_with_radix(&str[2..], 8, is_negative),
            "0x" => return parse_with_radix(&str[2..], 16, is_negative),
            _ => {}
        }
    }

    let value = match str {
        "inf" => Ok(f64::INFINITY),
         "nan" => Ok(f64::NAN),
        _ => match str.parse::<f64>() {
            Ok(n) => Ok(n),
            Err(_) => Err(ParseError::InvalidNumber(str.to_string())),
        },
    };

    value.and_then(|x| if is_negative {
        Ok(x * -1.0)
    } else {
        Ok(x)
    })
}

type SuffixValue = (&'static str, usize);
const SUFFIXES: [SuffixValue; 16] = [
    ("kib", 1024),
    ("ki",  1024),
    ("kb",  1000),
    ("k",   1000),
    ("mib", 1024 * 1024),
    ("mi",  1024 * 1024),
    ("mb",  1000 * 1000),
    ("m",   1000 * 1000),
    ("gib", 1024 * 1024 * 1024),
    ("gi",  1024 * 1024 * 1024),
    ("gb",  1000 * 1000 * 1000),
    ("g",   1000 * 1000 * 1000),
    ("tib", 1024 * 1024 * 1024 * 1024),
    ("ti",  1024 * 1024 * 1024 * 1024),
    ("tb",  1000 * 1000 * 1000 * 1000),
    ("t",   1000 * 1000 * 1000 * 1000)
];

#[inline]
pub fn get_number_suffix(s: &str) -> Option<&'static SuffixValue> {
    let lower = s.to_ascii_lowercase();
    SUFFIXES.iter().find(|x| {
        lower.ends_with(x.0)
    })
}

pub fn parse_number_with_unit(s: &str) -> ParseResult<f64> {
    let suffix = get_number_suffix(s);
    if let Some((ending, multiplier)) = suffix {
        let prefix = &s[0 .. s.len() - ending.len()];
        return match prefix.parse::<f64>() {
            Ok(n) => return Ok(n * (*multiplier as f64)),
            Err(_) => Err(ParseError::InvalidNumber(prefix.to_string()))
        }
    }
    Err(ParseError::InvalidNumber(s.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::lexer::number::parse_number_with_unit;

    #[test]
    fn test_parse_number_with_unit() {
        fn f(s: &str, expected: f64) {
            let v = parse_number_with_unit(s).unwrap();
            if v.is_nan() {
                assert!(expected.is_nan(),"unexpected value returned from parse_number_with_unit({}); got {}; want {}",
                        s, v, expected)
            } else {
                assert_eq!(v, expected,
                           "unexpected value returned from parse_number_with_unit({}); got {}; want {}",
                           s, v, expected)
            }
        }

        f("2k", (2 * 1000) as f64);
        f("2.3Kb", 2.3 * 1000_f64);
        f("3ki", (3 * 1024) as f64);
        f("4.5Kib", 4.5 * 1024_f64);
        f("2m", (2 * 1000 *1000) as f64);
        f("2.3Mb", 2.3 * 1000_f64 * 1000_f64);
        f("3Mi", (3_i64 * 1024 * 1024) as f64);
        f("4.5mib", 4.5 * (1024 * 1024) as f64);
        f("2G",     (2_i64 * 1000 * 1000 * 1000) as f64);
        f("2.3gB", 2.3 * (1000 *1000*1000) as f64);
        f("3gI", (3_i64 * 1024 * 1024 *1024) as f64);
        f("4.5GiB", 4.5 * (1024_i64 * 1024 *1024) as f64);
        f("2T", (2_i64 * 1000 * 1000 * 1000 * 1000) as f64);
        f("2.3tb", 2.3 * (1000_i64 * 1000 * 1000 *1000) as f64);
        f("3tI", (3 * 1024_i64 * 1024 * 1024 * 1024) as f64);
        f("4.5TIB", 4.5 * (1024_i64 * 1024 * 1024 * 1024) as f64)
    }
}