use enquote::{enquote, unescape, unquote};
use std::str;
use crate::parser::{ParseError, ParseResult};


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


/// extract_string_value interprets token as a single-quoted, double-quoted, or backquoted
/// Prometheus query language string literal, returning the string value that s
/// quotes.
///
/// Special-casing for single quotes was removed and single quoted strings are now treated the
/// same as double quoted ones.
pub fn extract_string_value(token: &str) -> ParseResult<String> {
    let n = token.len();

    if n < 2 {
        return Err(
            ParseError::General(
                format!("invalid quoted string literal. A minimum of 2 chars needed; got {}", token)
            ));
    }

    // See https://prometheus.io/docs/prometheus/latest/querying/basics/#string-literals
    let mut quote_ch = token.chars().next().unwrap();
    if !['"', '\'', '`'].contains(&quote_ch) {
        return Err(ParseError::General(format!("invalid quote character {}", quote_ch)))
    }

    let last = token.chars().last().unwrap();

    if last != quote_ch {
        return Err(
            ParseError::General(
                format!("string literal contains unexpected trailing char; got {}", token)
            ));
    }

    if n == 2 {
        return Ok("".to_string());
    }

    let mut s = &token[1 .. n-1];

    if quote_ch == '`' {
        if s.contains('`') {
            return Err(ParseError::General("invalid syntax".to_string()))
        }
        return Ok(s.to_string())
    }


    if quote_ch != '"' && quote_ch != '\'' {
        return Err(
            ParseError::General(
                format!("invalid quote character {}", quote_ch)
            ));
    }

    if s.contains('\n') {
        return Err(ParseError::General("Unexpected newline in string literal".to_string()));
    }

    if quote_ch == '\'' {
        let tok = s.replace("\\'", "'").replace( "\"", r#"\""#);
        s = tok.as_str();
        quote_ch = '"';
        return handle_unquote(s, quote_ch)
    }

    // Is it trivial? Avoid allocation.
    if !s.contains(&['\\', quote_ch]) {
        return Ok(s.to_string())
    }

    handle_unquote(s, quote_ch)
}

fn is_string_prefix(s: &str) -> bool {
    if s.is_empty() {
        return false
    }
    let ch = s.chars().next().unwrap();
    // See https://prometheus.io/docs/prometheus/latest/querying/basics/#string-literals
    ['"', '\'', '`'].contains(&ch)
}

#[inline]
fn handle_unquote(token: &str, quote: char) -> ParseResult<String> {
    match unescape(token, Some(quote)) {
        Err(err) => {
            let msg = format!("cannot parse string literal {}: {:?}", token, err);
            return Err(ParseError::General(msg));
        }
        Ok(s) => Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::unescape_ident;

    #[test]
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
