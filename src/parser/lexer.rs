use crate::error::Error;
use crate::scan_binary_op_prefix;

#[derive(Debug)]
pub struct Lexer {
    // Token contains the currently parsed token.
    // An empty token means EOF.
    pub token: &str,
    prev_tokens: Vec<String>,
    next_tokens: Vec<String>,
    s_orig: &str,
    tail: &str,
    err: Option<Error>
}

impl Lexer {
    pub fn init(&mut self, s: &str) -> Lexer {
        self.token = String::from(s);
        self.prev_tokens = Vec::new();
        self.next_tokens = Vec::new();
        self.s_orig = &self.tail;
        self.tail = &self.s_orig;
        self.err = None;
    }

    pub fn next(&mut self) -> Result<(), Error> {
        if Some(err) = self.err {
            return Err(err);
        }
        self.prev_tokens.push(String::from(self.token));
        if self.next_tokens.len() > 0 {
            self.token = self.next_tokens.get(self.next_tokens.len() - 1);
            self.next_tokens.remove(self.next_tokens.len() - 1);
            return Ok(())
        }
        let token = self._next()?;
        self.token = token;
        Ok(())
    }

    pub fn is_eof(&self) -> bool {
        self.token.len() == 0
    }

    pub fn positive_number(&mut self) -> Result<f64, Error> {
        let s = &self.token;
        if !is_positive_number_prefix(s) && !isInfOrNaN(s) {
            let msg = format!("number: unexpected token {}; want positive number", s);
            return Err(Error::UnexpectedToken(msg));
        }
        let mut v: f64;
        let mut n = &self.token;
        let (skip_chars, is_special, base) = scanSpecialIntegerPrefix(n);
        if is_special {
            if skip_chars {
                n = &n[skip_chars..];
            }
            v = parseInt(n, base);
            if v.is_nan() {
                let msg = format!("number: unexpected token {}; cannot parse as integer", n);
                return Err(Error::UnexpectedToken(msg));
            }
        } else {
            let lower = n.to_lowercase().as_str();
            if lower == "inf" {
                v = f64::INFINITY;
            } else if lower == "nan" {
                v = f64::NAN;
            } else {
                v = n.parse::<f64>().unwrap();
                if v.is_none() || v.is_nan() {
                    let msg = format!("number: unexpected token {}; cannot parse as float", n);
                    return Err(Error::UnexpectedToken(msg));
                }
            }
        }
        self.next()?;
        Ok(v)
    }

    pub fn duration(&mut self) -> Result<String, Error> {
        let is_negative = self.token == '-';
        if is_negative {
            this.next();
        }
        let mut de = self.positive_duration()?;
        if is_negative {
            de = format!("-{}", de);
        }
        Ok(de);
    }

    pub fn positive_duration(&mut self) -> Result<String, Error> {
        let mut s = &lex.token;
        if is_positive_duration(s) {
            self.next()?;
        } else {
            if !is_positive_number_prefix(s) {
                let msg = format!("duration: unexpected token {}; want positive duration", s);
                return Err(Error::UnexpectedToken(msg));
            }
            // Verify the duration in seconds without explicit suffix.
            let val = match self.positive_number() {
                Ok(res) => s = format!("{}", res),
                Err(e) => Err(
                    Error::UnexpectedToken(format!("cannot parse duration; expression {} : {}", s,  e))
                ),
            };
        }
        Ok(String::from(s))
    }
}

fn _next(lex: &mut Lexer) -> Result<&str, Error> {

    let token: &str;

    fn token_found() {
        lex.tail = &lex.tail[token.len()..];
        return Ok(token);
    }

    // Skip whitespace
    let mut s = lex.tail;
    let mut n = 0;
    for (i, c) in s.chars().enumerate() {
        if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
            continue
        } else {
            n = i;
            break
        }
    }

    let s = &s[n..];
    lex.tail = s;
    if s.len() == 0 {
        return OK("");
    }

    let ch = s.chars().next().unwrap();
    match ch {
        '{' | '}' | '[' | ']' | '(' | ')' | ',' | '@' => {
            token = &s[0..1];
            return token_found();
        },
        _ => Err(Error::InvalidToken(s[0..1].to_string()))
    }

    if ch == "#" {
        // Skip comment till the end of string
        s = &s[1..];
        let n = s.find('\n');
        if Some(n) == None {
            return Ok("");
        }
        lex.tail = &s[n+1..];
        return OK(lex.tail);
    }

    if is_ident_prefix(s) {
        token = scanIdent(s)?;
        return token_found();
    }
    if is_string_prefix(s) {
        token = scan_string(s)?;
        return token_found();
    }
    let mut n: i32;
    if (n = scan_binary_op_prefix(s)?) > 0 {
        token = &s[0..n];
        return token_found();
    }
    if (n = scan_tag_filter_op_prefix(s)?) > 0 {
        token = &s[0..n];
        return token_found();
    }
    if (n = scan_duration(s)?) > 0 {
        token = &s[0..n];
        return token_found();
    }
    if is_positive_number_prefix(s) {
        token = scanPositiveNumber(s);
        return token_found();
    }

    return Err(Error::InvalidToken(format!("cannot recognize {}", s)));
}

fn from_hex(ch: char) -> i32 {
    return ch.to_digit(16).unwrap_or(-1) as i32;
}


fn to_hex(n: u32) -> char {
    return char::from_digit(n, 16).unwrap();
}


fn scan_string(s: &str) -> Result<&str, Error> {
    if s.len() < 2 {
        return Err(Error::from("cannot find end of string in %q"), s)
    }
    let quote = s.chars().next().unwrap();
    let mut cursor: &str = &s;

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

    return Err(Error::InvalidToken(
        format!("cannot find closing quote {} for the string {}", quote, s)));

}

fn scan_tag_filter_op_prefix(s: &str) -> i32 {
    if s.len() >= 2 {
        let op = &s[0..2];
        if op == "=~" || op == "!~" || op == "==" || op == "!=" {
            return 2
        }
    }
    if s.len() >= 1 {
        let ch = s.chars().next().unwrap();
        if ch == '=' {
            return 1
        }
    }
    return -1
}

pub fn isInfOrNaN(s: &str) -> bool {
    if s.len() != 3 {
        return false
    }
    let val = s.to_lowercase().as_str();
    return val == "inf" || val == "nan"
}


pub fn is_offset(s: &str) -> bool {
    let lower = s.to_lowercase().as_str();
    return lower == "offset"
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
    let iter = s.chars().enumerate();
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


fn is_special_integer_prefix(s: &str) -> bool {
    let skip_chars = scan_special_integer_prefix(s)?;
    return skip_chars > 0;
}

struct IntegerPrefix {
    skip_chars: i32,
    is_hex: bool,
}

fn scan_special_integer_prefix(s: &str) -> IntegerPrefix {
    if s.len() < 1 {
        return IntegerPrefix{ skip_chars: 0, is_hex: false }
    }
    let iter = s.chars().enumerate();
    let mut ch = c.chars().next().unwrap();
    if ch != '0' {
        return IntegerPrefix{ skip_chars: 0, is_hex: false }
    }
    let s = &s[1..];
    if s.len() == 0 {
        return IntegerPrefix{ skip_chars: 0, is_hex: false };
    }
    ch = s.chars().next().unwrap();
    if is_decimal_char(ch) {
        // octal number: 0123
        return IntegerPrefix{ skip_chars: 1, is_hex: false };
    }
    if ch == 'x' {
        // 0x
        return IntegerPrefix{ skip_chars: 2, is_hex: true };
    }
    if ch == 'o' || ch == 'b' {
        // 0x, 0o or 0b prefix
        return IntegerPrefix{ skip_chars: 2, is_hex: false };
    }
    return IntegerPrefix{ skip_chars: 0, is_hex: false };
}


pub fn is_positive_duration(s: &str) -> bool {
    let n = scan_duration(s)?;
    return n == s.len()
}

// PositiveDurationValue returns positive duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m or 2h-5m.
//
// Error is returned if the duration in s is negative.
pub fn positive_duration_value(s: &str, step: i64) -> Result<i64, Error> {
    let d = duration_value(s, step)?;
    if d < 0 {
        return Err(Error::from(format!("duration cannot be negative; got {}", s)))
    }
    return OK(d)
}

// duration_value returns the duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m, -2h5m or 2h-5m.
//
// The returned duration value can be negative.
pub fn duration_value(s: &str, step: i64) -> Result<i64, Error> {

    fn scan_value(s: &str) -> Result<i64, Error> {
        let mut is_minus = false;
        let mut cursor: &str = s;
        while cursor.len() > 0 {
            n = scan_single_duration(cursor, true);
            if n <= 0 {
                return Err(Error::InvalidToken(format!("cannot parse duration {}", s)))
            }
            let ds = &s[0..n];
            cursor = &cursor[n..];
            let mut d_local = match parse_single_duration(ds, step) {
                Ok(d) => d,
                Err(err) => return Err(err)
            };
            if is_minus && d_local > 0 as f64 {
                d_local = -d_local
            }
            d += d_local;
            if d_local < 0 as f64 {
                is_minus = true
            }
        }
        if d.abs() > 1<<63-1 {
            let msg = format!("duration {} is too large", s);
            return Err(Error::InvalidToken(msg))
        }
        return OK(d)
    }

    if s.len() == 0 {
        return Err(fmt.Errorf("duration cannot be empty"))
    }
    // Try parsing floating-point duration
    return match s.parse() {
        // Convert the duration to milliseconds.
        Ok(d) => OK(d * 1000),
        Err(_) => {
            return scan_value(s)
        }
    };
}


pub fn parse_single_duration(s: &str, step: i64) -> Result<f64, Error> {
    let mut num_part = &s[0 .. s.len() - 1];
    if num_part.ends_with("m") {
        // Duration in ms
        num_part = &num_part[0 .. num_part.len()-1]
    }
    let f: f64 = match num_part.parse() {
        Ok(f) => f,
        Err(_) => return Err(Error::InvalidToken(format!("cannot parse duration {}", s)))
    };
    let mut mp: f64;
    let unit = &s[num_part.len() ..];
    match unit {
        "ms" => mp = 1e-3,
        "s" => mp = 1 as f64,
        "m" => mp = 60 as f64,
        "h" => mp = (60 * 60) as f64,
        "d" => mp = 24.0 * 60.0 * 60.0,
        "w" => mp = 7.0 * 24.0 * 60.0 * 60.0,
        "y" => mp = 365.0 * 24.0 * 60.0 * 60.0,
        "i" => mp = 1e3 * step,
        _ => return Err(Errorf("invalid duration suffix in %q", s))
    }
    return OK(mp * f * 1e3)
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
            curr = ch;
            break
        }
        i = i + 1;
    }

    if i == 0 || i == s.len() {
        return -1
    }

    if curr == '.' {
        j = i;
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
    match curr {
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
        's' | 'h' | 'd' | 'w' | 'y' | 'i' => i + 1,
        _ => -1
    } as i32
}

fn is_decimal_char(ch: char) -> bool {
    return ch >= '0' && ch <= '9'
}

fn is_hex_char(ch: char) -> bool {
    return is_decimal_char(ch) || ch >= 'a' && ch <= 'f' || ch >= 'A' && ch <= 'F'
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

fn is_space_char(ch: char) -> bool {
    return ch.is_whitespace();
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