use std::str;
use enquote::{enquote, unquote};
use crate::error::{Error, Result};
use crate::lexer::scan_duration;

pub fn scan_string(s: &str) -> Result<&str> {
    if s.len() < 2 {
        return Err(Error::new(format!("cannot find end of string in {}", s)));
    }
    let quote = s.chars().next().unwrap();
    let mut cursor: &str = &s;

    let mut skip_delimiter = false;

    for (i, ch) in cursor.char_indices() {
        if !skip_delimiter {
            if ch == '\\' {
                skip_delimiter = true;
            } else if ch == quote {
                let token = &s[0 .. i+1];
                return Ok(token);
            }
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
    return is_first_ident_char(&ch);
}

#[inline]
fn is_first_ident_char(ch: &char) -> bool {
    match ch {
        'A'..='Z' | 'a'..='z' => {
            true
        },
        '_' | ':' => true,
        _ => false
    }
}

fn is_ident_char(ch: &char) -> bool {
    match ch {
        'A'..='Z' | 'a'..='z' | '0' ..='9' => {
            true
        },
        '_' | ':' | '.' => true,
        _ => false
    }
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

pub fn unescape_ident(str: &str) -> String {
    unquote(str).unwrap()
}

pub fn quote(str: &str) -> String {
    enquote('\"', str)
}