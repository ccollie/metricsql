use enquote::{enquote, unquote};
use std::str;

pub fn is_string_prefix(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let ch = s.chars().next().unwrap();
    matches!(ch, '"' | '\'' | '`')
}

#[inline]
fn is_first_ident_char(ch: &char) -> bool {
    matches!(ch, 'A'..='Z' | 'a'..='z' | '_' | ':')
}

fn is_ident_char(ch: char) -> bool {
    matches!(ch, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | ':' | '.')
}

pub fn escape_ident(s: &str) -> String {
    let mut dst = String::new();
    for (i, ch) in s.chars().enumerate() {
        if is_ident_char(ch) {
            if i == 0 && !is_first_ident_char(&ch) {
                // hex escape the first char
                dst.push_str("\\x");
                dst.push_str(&*format!("{:02x}", ch as u8).to_string());
            } else {
                dst.push(ch);
            }
            continue;
        } else {
            // escape the char
            dst.push(ch.escape_default().next().unwrap());
        }
    }
    dst.to_string()
}

pub fn unescape_ident(str: &str) -> String {
    unquote(str).unwrap()
}

pub fn quote(str: &str) -> String {
    enquote('\"', str)
}
