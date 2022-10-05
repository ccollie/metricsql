use enquote::{enquote, unescape};
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
    unescape(str, None).unwrap()
}

pub fn quote(str: &str) -> String {
    enquote('\"', str)
}

#[cfg(test)]
mod tests {
    use crate::lexer::unescape_ident;

    #[inline]
    fn test_unescape_ident() {
        fn f(s: &str, expected: &str) {
            let result = unescape_ident(s);
            assert_eq!(result, expected, "unexpected result for unescape_ident({}); got {}; want {}", s, result, expected)
        }

        f("", "");
        f("a", "a");
        f("\\", "");
        f(r"\\", "\\");
        f(r"\foo\-bar", "foo-bar");
        f(r#"a\\\\b\"c\d"#, r#"a\\b"cd"#);
		f(r"foo.bar:baz_123#", r"foo.bar:baz_123");
        f(r"foo\ bar", "foo bar");
        f(r"\x21", "!");
        f(r"\xeDfoo\x2Fbar\-\xqw\x", r"\xedfoo\x2fbar-xqwx");
        f(r"\п\р\и\в\е\т123", "привет123")
    }
}
