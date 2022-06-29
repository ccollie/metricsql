use crate::error::Error;
use crate::lexer::scan_duration;

pub fn scan_string(s: &str) -> Result<&str, Error> {
    if s.len() < 2 {
        return Err(Error::new(format!("cannot find end of string in {}", s)));
    }
    let quote = s.chars().next().unwrap();
    let mut cursor: &str = &s;

    let mut skip_delimiter = false;

    for (i, ch) in cursor.char_indices() {
        if ch == '\\' && !skip_delimiter {
            skip_delimiter = true;
        } else if ch == quote && !skip_delimiter {
            let token = &s[0 .. i+1];
            return Ok(token);
        } else {
            skip_delimiter = false;
        }
    }

    return Err(Error::from(
        format!("cannot find closing quote {} for the string {}", quote, s)));

}

pub fn is_string_prefix(s: &str) -> bool {
    if s.len() == 0 {
        return false
    }
    let ch = s.chars().next().unwrap();
    match ch {
        // See https://prometheus.io/docs/prometheus/latest/querying/basics/#string-literals
        '"' | '\'' | '`' => true,
        _ => false
    }
}

pub(crate) fn is_positive_number_prefix(s: &str) -> bool {
    if s.len() == 0 {
        return false
    }
    let mut ch = s.chars().next().unwrap();
    if is_decimal_char(ch) {
        return true;
    }
    ch = s.chars().next().unwrap();
    // Check for .234 numbers
    if ch != '.' || s.len() < 2 {
        return false;
    }
    return is_decimal_char(ch);
}


pub fn is_positive_duration(s: &str) -> bool {
    let n = scan_duration(s)?;
    return n == s.len()
}

fn is_decimal_char(ch: char) -> bool {
    return ch >= '0' && ch <= '9'
}

pub fn is_ident_prefix(s: &str) -> bool {
    if s.len() == 0 {
        return false;
    }
    let ch = s.chars().next().unwrap();
    if ch == '\\' {
        // Assume this is an escape char for the next char.
        return true;
    }
    return is_first_ident_char(ch);
}

fn is_first_ident_char(ch: char) -> bool {
    if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' {
        return true;
    }
    return ch == '_' || ch == ':';
}

fn is_ident_char(ch: char) -> bool {
    if is_first_ident_char(ch) {
        return true;
    }
    return is_decimal_char(ch) || ch == '.';
}

pub fn append_escaped_ident(dst: &str, s: &str) -> &str {
    let escaped = escape_ident(s);
    dst.push_str(escaped);
    return dst
}

pub fn escape_ident(s: &str) -> &str {
    let mut dst = String::new();
    for (i, ch) in s.chars().enumerate() {
        if is_ident_char(*ch) {
            if i == 0 && !is_first_ident_char(*ch) {
                // hex escape the first char
                dst.push_str("\\x");
                dst.push_str(format!("{:02x}", *ch as u8).as_slice());
            } else {
                dst.push(*ch)
            }
            continue;
        } else {
            // escape the char
            dst.push(ch.escape_default().next().unwrap());
        }
    }
    return dst.as_str()
}