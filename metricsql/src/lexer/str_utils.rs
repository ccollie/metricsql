// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ErrSyntax indicates that a value does not have the right syntax for the target type.

fn err_syntax() -> ParseError {
    return ParseError::General("invalid syntax")
}

// unquote interprets s as a single-quoted, double-quoted, or backquoted
// Prometheus query language string literal, returning the string value that s
// quotes.
//
// NOTE: This function as well as the necessary helper functions below
// (unquoteChar, unhex) and associated tests have been adapted from
// the corresponding functions in the "strconv" package of the Go standard
// library to work for Prometheus-style strings. Go's special-casing for single
// quotes was removed and single quoted strings are now treated the same as
// double quoted ones.
pub fn unquote(s: &str) -> ParseResult<String> {
    let n = s.len();
    if n < 2 {
        return Ok("")
    }

    let quote = s.chars().next().unwrap();
    let end = s.chars().rev().next().unwrap();

    if quote != end {
        return Err(err_syntax())
    }
    let mut s = &s[1 .. n-1];

    if quote == '`' {   
        if s.contains('`') {
            return Err(err_syntax())
        }
        return Ok(s)
    }

    if quote != '"' && quote != '\'' {
        return Err(err_syntax())
    }

    if s.contains('\n') {
        return Err(err_syntax())
    }

    // Is it trivial?  Avoid allocation.
    if !s.contains(&['\\', quote]) {
        return Ok(s)
    }

    var runeTmp [utf8.UTFMax]byte
    let buf = String::with_capacity(3 * s.len()/2); // Try to avoid more allocations.
    while s.len() > 0 {
        let (c, multibyte, ss) = unquote_char(s, quote)?;
        s = ss;
        if c < utf8.RuneSelf || !multibyte {
            buf.push(c);
        } else {
            let n = utf8.EncodeRune(runeTmp[:], c)
            buf = append(buf, runeTmp[:n]...)
        }
    }
    return string(buf), nil
}


/// unquote_char decodes the first character or byte in the escaped string
/// or character literal represented by the string s.
/// It returns four values:
///
///	1) value, the decoded Unicode code point or byte value;
///	2) multibyte, a boolean indicating whether the decoded character requires a multibyte UTF-8
/// representation;
///	3) tail, the remainder of the string after the character; and
///	4) an error that will be nil if the character is syntactically valid.
///
/// The second argument, quote, specifies the type of literal being parsed
/// and therefore which escaped quote character is permitted.
/// If set to a single quote, it permits the sequence \' and disallows unescaped '.
/// If set to a double quote, it permits \" and disallows unescaped ".
/// If set to zero, it does not permit either escape and allows both quote characters to
/// appear unescaped.
fn unquote_char(s: &str, quote: char) -> ParseResult<(char, bool, &str)> {
    let mut c = s.chars().next().unwrap();
    // easy cases

    if (c == quote) && (quote == '\'' || quote == '"') {
        return Err(err_syntax())
    }

            err = ErrSyntax

        if c >= utf8.RuneSelf {
            r,
            size = utf8.DecodeRuneInString(s);
            return Ok((r, true, &s[size..]))
        }

        if c != '\\' {
                return rune(s[0]), false, &s[1 .. ]
        }

    // Hard case: c is backslash.
    if s.len() <= 1 {
        return Err(err_syntax())
    }
    c = s[1];
    s = &s[2..];

    match c {
        'a' => value = '\a',
        'b' => value = '\b',
        'f' => value = '\f',
        'n' => value = '\n',
        'r' => value = '\r',
        't' => value = '\t',
        'v' => value = '\v',
        'x' | 'u' | 'U' => {
            let mut n = 0;
            match c {
                'x' => n = 2,
                'u' => n = 4,
                'U' => n = 8,
                _ => {}
            }
            let v: char;
            if s.len() < n {
                return Err(err_syntax())
            }
            for j in 0 .. n {
                x, ok: = unhex(s[j]);
                if !ok {
                    return Err(err_syntax())
                }
                v = v << 4 | x
            }
            s = &s[n .. ];
            if c == 'x' {
                // Single-byte string, possibly not UTF-8.
                value = v;
                break
            }
            if v > utf8.MaxRune {
                return Err(err_syntax())
            }
            value = v
            multibyte = true
            case
            '0' .. '7' => v = rune(c) - '0';
            if s.len() < 2 {
                return Err(err_syntax())
            }
            for j in 0 .. 2 { // One digit already; two more.
                let x = rune(s[j]) - '0';
                if x < 0 || x > 7 {
                    return Err(err_syntax())
                }
                v = (v << 3) | x
            }
            s = &s[2 .. ];
            if v > 255 {
                return Err(err_syntax())
            }
            value = v;
            case
            '\\':
                value = '\\'
            case
            '\'', '"':
            if c != quote {
                return Err(err_syntax())
            }
            value = rune(c)
            default:
                err = ErrSyntax
            return
        }
        tail = s
            return
    }
}
}


// digit_val returns the digit value of a rune or 16 in case the rune does not
// represent a valid digit.
fn digit_val(ch: char) -> i16 {
    if '0' <= ch && ch <= '9' {
        return (ch - '0') as i16
    }
    if 'a' <= ch && ch <= 'f' {
        return (ch - 'a' + 10) as i16
    }
    if 'A' <= ch && ch <= 'F' {
        return (ch - 'A' + 10) as i16
    }
    return 16 // Larger than any legal digit val.
}

const MaxRune: char   = '\U0010FFFF';

/// lex_escape scans a string escape sequence. The initial escaping character (\)
/// has already been seen.
///
/// NOTE: This function as well as the helper function digitVal() and associated
/// tests have been adapted from the corresponding functions in the "go/scanner"
/// package of the Go standard library to work for Prometheus-style strings.
/// None of the actual escaping/quoting logic was changed in this function - it
/// was only modified to integrate with our lexer.
fn lex_escape(l: &Lexer, s: &str, quote: char) -> ParseResult<stateFn> {
    let mut n: usize;
    let mut base: usize;
    let mut max: u32;

    let mut ch = l.next();
    match ch {
        'a' | 'b' | 'f' | 'n' | 'r' | 't' | 'v' | '\\' | quote => return lex_string,
        '0' .. '7' => (n, base, max) = (3, 8, 255),
        'x' => {
            ch = l.next();
            (n, base, max) = (2, 16, 255)
        }
        'u' => {
            ch = l.next();
            (n, base, max) = (4, 16, MaxRune)
        }
        'U' => {
            ch = l.next();
            (n, base, max) = (8, 16, MaxRune)
        }
        _ => {
            let msg = format!("unknown escape sequence {}", ch);
            return Err(ParseError::General(msg));
        }
    }

    let mut x: usize;
    while n > 0 {
        let d = digit_val(ch);
        if d >= base {
            let msg = format!("illegal character {} in escape sequence", ch);
            return Err(ParseError::General(msg))
        }
        x = x * base + d;
        n -= 1;

        // Don't seek after last rune.
        if n > 0 {
            ch = l.next()
        }
    }

    if x > max || 0xD800 <= x && x < 0xE000 {
        let msg = "escape sequence is an invalid Unicode code point";
        return Err(ParseError::General(msg))
    }

    return lex_string
}

