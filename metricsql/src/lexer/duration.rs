use crate::error::{Error, Result};

// positive_duration_value returns positive duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m or 2h-5m.
//
// Error is returned if the duration in s is negative.
pub fn positive_duration_value(s: &str, step: i64) -> Result<i64> {
    let d = duration_value(s, step)?;
    if d < 0 {
        return Err(Error::from(format!("duration cannot be negative; got {}", s)))
    }
    return Ok(d)
}

// duration_value returns the duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m, -2h5m or 2h-5m.
//
// The returned duration value can be negative.
pub fn duration_value(s: &str, &step: i64) -> Result<i64> {

    fn scan_value(s: &str, step: &i64) -> Result<i64> {
        let mut is_minus = false;
        let mut cursor: &str = s;
        let mut d = 0.0;

        while cursor.len() > 0 {
            let n = scan_single_duration(cursor, true);
            if n <= 0 {
                return Err(Error::new(format!("cannot parse duration {}", s)))
            }
            let ds = &s[0..n];
            cursor = &cursor[n..];
            let mut d_local = parse_single_duration(ds, step)?;
            if &is_minus && &(d_local > 0.0) {
                d_local = -d_local
            }
            d += d_local;
            if d_local < 0 as f64 {
                is_minus = true
            }
        }
        if d.abs() > (1 << 63 - 1) as f64 {
            let msg = format!("duration {} is too large", s);
            return Err(Error::new(msg))
        }
        return Ok(d)
    }

    if s.len() == 0 {
        return Err(Error::new("duration cannot be empty"))
    }
    // Try parsing floating-point duration
    return match s.parse() {
        // Convert the duration to milliseconds.
        Ok(d) => Ok(d * 1000),
        Err(_) => {
            return scan_value(s, step)
        }
    };
}


pub fn parse_single_duration(s: &str, &step: &i64) -> Result<f64> {
    let mut num_part = &s[0 .. s.len() - 1];
    if num_part.ends_with("m") {
        // Duration in ms
        num_part = &num_part[0 .. num_part.len()-1]
    }
    let f: f64 = match num_part.parse() {
        Ok(f) => f,
        Err(_) => return Err(Error::new(format!("cannot parse duration {}", s)))
    };
    let mut mp: f64;
    let unit = &s[num_part.len() ..];
    match unit {
        "ms" => mp = 1e-3,
        "s" => mp = 1.0 as f64,
        "m" => mp = 60.0 as f64,
        "h" => mp = (60.0 * 60.0) as f64,
        "d" => mp = 24.0 * 60.0 * 60.0,
        "w" => mp = 7.0 * 24.0 * 60.0 * 60.0,
        "y" => mp = 365.0 * 24.0 * 60.0 * 60.0,
        "i" => mp = 1e3 * step,
        _ => return Err(Error::from(format!("invalid duration suffix in {}", s)))
    }
    return Ok(mp * f * 1e3)
}

// scan_duration scans duration, which must start with positive num.
//
// I.e. 123h, 3h5m or 3.4d-35.66s
pub fn scan_duration(s: &str) -> i32 {
    // The first part must be non-negative
    let mut n = scan_single_duration(s, false);
    if n <= 0 {
        return -1
    }
    let mut cursor: &str = &s[n..];
    let mut i = n;
    loop {
        // Other parts may be negative
        n = scan_single_duration(s, true);
        if n <= 0 {
            return i
        }
        cursor = &cursor[n..];
        i += n
    }
}


fn scan_single_duration(s: &str, can_be_negative: bool) -> i32 {
    if s.len() == 0 {
        return -1
    }
    let mut i = 0;

    let mut ch = s.chars().next().unwrap();
    if ch == '-' && can_be_negative {
        i = i + 1;
    }

    let mut cursor = &s[i..];
    let mut curr: char = ch;

    for (_k, ch) in cursor.chars().enumerate() {
        if !is_decimal_char(ch) {
            curr = ch.clone();
            break
        }
        i = i + 1;
    }

    if i == 0 || i == s.len() {
        return -1
    }

    if curr == '.' {
        let mut j = i.clone();
        i = i + 1;
        cursor = &s[i..];
        for c in cursor.chars().enumerate() {
            if !is_decimal_char(*c) {
                curr = *c;
                break
            }
            i = i + 1;
        }
        if i == j || i == s.len() {
            return -1
        }
    }
    return match curr {
        'm' => {
            if i + 1 < s.len() {
                cursor = &s[i..];
                curr = cursor.chars().next().unwrap();
                if curr == 's' {
                    // duration in ms
                    i = i + 2;
                }
            }
            // duration in minutes
            return (i + 1) as i32;
        },
        's' | 'h' | 'd' | 'w' | 'y' | 'i' => (i + 1) as i32,
        _ => -1
    }
}

fn is_decimal_char(ch: char) -> bool {
    return ch >= '0' && ch <= '9'
}