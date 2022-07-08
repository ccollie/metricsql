use crate::parser::ParseError;

/// duration_value returns the duration in milliseconds for the given s
/// and the given step.
///
/// Duration in s may be combined, i.e. 2h5m, -2h5m or 2h-5m.
///
/// The returned duration value can be negative.
pub fn duration_value(s: &str, step: i64) -> Result<i64, ParseError> {
    fn scan_value(s: &str, step: i64) -> Result<i64, ParseError> {
        let mut is_minus = false;
        let mut cursor: &str = s;
        let mut d = 0.0;

        while !cursor.is_empty() {
            let n = scan_single_duration(cursor, true);
            if n <= 0 {
                return Err(ParseError::InvalidDuration(s.to_string()));
            }
            let ds = &s[0..n as usize];
            cursor = &cursor[n as usize..];
            let mut d_local = parse_single_duration(ds, &step)?;
            if is_minus && (d_local > 0.0) {
                d_local = -d_local
            }
            d += d_local;
            if d_local < 0 as f64 {
                is_minus = true
            }
        }
        if d.abs() > (1_i64 << (62 - 1)) as f64 {
            let msg = format!("duration {} is too large", s);
            return Err(ParseError::General(msg));
        }
        Ok(d as i64)
    }

    if s.is_empty() {
        return Err(ParseError::General("duration cannot be empty".to_string()));
    }
    // Try parsing floating-point duration
    match s.parse::<i64>() {
        // Convert the duration to milliseconds.
        Ok(d) => Ok(d * 1000),
        Err(_) => scan_value(s, step),
    }
}

pub fn parse_single_duration(s: &str, &step: &i64) -> Result<f64, ParseError> {
    let mut num_part = &s[0..s.len() - 1];
    if num_part.ends_with('m') {
        // Duration in ms
        num_part = &num_part[0..num_part.len() - 1]
    }
    let f: f64 = match num_part.parse() {
        Ok(f) => f,
        Err(_) => return Err(ParseError::InvalidDuration(s.to_string())),
    };
    let mp: f64;
    let unit = &s[num_part.len()..];
    match unit {
        "ms" => mp = 1e-3,
        "s" => mp = 1.0_f64,
        "m" => mp = 60.0_f64,
        "h" => mp = (60.0 * 60.0) as f64,
        "d" => mp = 24.0 * 60.0 * 60.0,
        "w" => mp = 7.0 * 24.0 * 60.0 * 60.0,
        "y" => mp = 365.0 * 24.0 * 60.0 * 60.0,
        "i" => mp = 1e3 * step as f64,
        _ => {
            return Err(ParseError::General(format!(
                "invalid duration suffix in {}",
                s
            )))
        }
    }
    Ok(mp * f * 1e3)
}

// scan_duration scans duration, which must start with positive num.
//
// I.e. 123h, 3h5m or 3.4d-35.66s
pub fn scan_duration(s: &str) -> i32 {
    // The first part must be non-negative
    let mut n = scan_single_duration(s, false);
    if n <= 0 {
        return -1;
    }

    let mut cursor: &str = &s[n as usize..];
    let mut i = n;
    loop {
        // Other parts may be negative
        n = scan_single_duration(s, true);
        if n <= 0 {
            return i;
        }
        cursor = &cursor[n as usize..];
        i += n
    }
}

fn scan_single_duration(s: &str, can_be_negative: bool) -> i32 {
    if s.is_empty() {
        return -1;
    }
    let mut i = 0;

    let ch = s.chars().next().unwrap();
    if ch == '-' && can_be_negative {
        i += 1;
    }

    let mut cursor = &s[i..];
    let mut curr: char = ch;

    for (_k, ch) in cursor.chars().enumerate() {
        if !is_decimal_char(ch) {
            curr = ch;
            break;
        }
        i += 1;
    }

    if i == 0 || i == s.len() {
        return -1;
    }

    if curr == '.' {
        let j = i;
        i += 1;
        cursor = &s[i..];
        for c in cursor.chars() {
            if !is_decimal_char(c) {
                curr = c;
                break;
            }
            i += 1;
        }
        if i == j || i == s.len() {
            return -1;
        }
    }
    return match curr {
        'm' => {
            if i + 1 < s.len() {
                cursor = &s[i..];
                curr = cursor.chars().next().unwrap();
                if curr == 's' {
                    // duration in ms
                    i += 2;
                }
            }
            // duration in minutes
            return (i + 1) as i32;
        }
        's' | 'h' | 'd' | 'w' | 'y' | 'i' => (i + 1) as i32,
        _ => -1,
    };
}

fn is_decimal_char(ch: char) -> bool {
    ('0'..='9').contains(&ch)
}
