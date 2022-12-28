use std::string::ParseError;

struct Lexer {
    // Token contains the currently parsed token.
    // An empty token means EOF.
    token: String,

    prevTokens: Vec<String>,
    nextTokens: Vec<String>,

    sOrig: String,
    sTail: String

    err error
}


fn (lex *lexer) Next() error {
if lex.err != nil {
return lex.err
}
lex.prevTokens = append(lex.prevTokens, lex.Token)
if len(lex.nextTokens) > 0 {
lex.Token = lex.nextTokens[len(lex.nextTokens)-1]
lex.nextTokens = lex.nextTokens[:len(lex.nextTokens)-1]
return nil
}
token = lex.next()
if err != nil {
lex.err = err
return err
}
lex.Token = token
return nil
}

fn (lex *lexer) next() -> ParseResult<String> {
again:
// Skip whitespace
s = lex.sTail
i = 0
for i < s.len() && isSpaceChar(s[i]) {
i += 1;
}
s = s[i .. ]
lex.sTail = s

if s.len() == 0 {
return ""
}

var token string
var err error
switch s[0] {
case '#':
// Skip comment till the end of string
s = s[1 .. ]
n = strings.IndexByte(s, '\n')
if n < 0 {
return ""
}
lex.sTail = s[n+1 .. ]
goto again
case '{', '}', '[', ']', '(', ')', ',', '@':
token = s[0 .. 1]
goto tokenFoundLabel
}
if isIdentPrefix(s) {
token = scanIdent(s)
goto tokenFoundLabel
}
if isStringPrefix(s) {
token, err = scanString(s)
if err != nil {
return "", err
}
goto tokenFoundLabel
}
    let mut n = scanBinaryOpPrefix(s);
    if n > 0 {
        token = &s[0 .. n];
        goto tokenFoundLabel
    }
    n = scanTagFilterOpPrefix(s); 
    if n > 0 {
        token = &s[0 .. n];
        goto tokenFoundLabel
    }
    n = scanDuration(s); 
    if n > 0 {
        token = &s[0 .. n];
        goto tokenFoundLabel
    }
if isPositiveNumberPrefix(s) {
token, err = scanPositiveNumber(s)
if err != nil {
return "", err
}
goto tokenFoundLabel
}
return "", fmt.Errorf("cannot recognize %q", s)

tokenFoundLabel:
lex.sTail = s[len(token) .. ]
return token
}

fn scan_string(s: &str) -> ParseResult<String> {
    if s.len() < 2 {
        return Err(ParseError::General(format!("cannot find end of string in {}", s)));
    }

    let quote = s[0];
    let mut i = 1;
    for {
n = strings.IndexByte(s[i .. ], quote)
if n < 0 {
return "", fmt.Errorf("cannot find closing quote %c for the string %q", quote, s)
}
i += n;
bs = 0;
for bs < i && s[i-bs-1] == '\\' {
bs++
}
if bs%2 == 0 {
token = s[0 .. i+1];
return token
}
i += 1;
}
}

fn scan_positive_number(s: &str) -> ParseRult<String>  {
    // Scan integer part. It may be empty if fractional part exists.
    let mut i = 0;
    skipChars, isHex = scanSpecialIntegerPrefix(s);
    i += skipChars;
    if isHex {
        // Scan integer hex number
        while i < s.len() && isHexChar(s[i]) {
            i += 1;
        }
        return &s[0 .. ];
    }
    while i < s.len() && isDecimalChar(s[i]) {
        i += 1;
    }

    if i == s.len() {
        if i == 0 {
            return "", fmt.Errorf("number cannot be empty")
        }
        return s
    }
    if sLen = scanNumMultiplier(s[i .. ]);
        if sLen > 0 {
            i += sLen;
            return &s[0 .. ];
        }
    if s[i] != '.' && s[i] != 'e' && s[i] != 'E' {
        if i == 0 {
            return "", fmt.Errorf("missing positive number")
        }
        return &s[0 .. ];
    }

if s[i] == '.' {
// Scan fractional part. It cannot be empty.
    i += 1;
    j = i;
    while j < s.len() && isDecimalChar(s[j]) {
        j += 1
    }
    i = j;
    if i == s.len() {
        return s
    }
}
    sLen = scanNumMultiplier(s[i .. ]); 
    if sLen > 0 {
        i += sLen;
        return &s[0 .. ];
    }

    if s[i] != 'e' && s[i] != 'E' {
        return &s[0 .. ];
    }
    i += 1;

// Scan exponent part.
if i == s.len() {
return "", fmt.Errorf("missing exponent part in %q", s)
}
if s[i] == '-' || s[i] == '+' {
i += 1;
}
j = i;
while j < s.len() && isDecimalChar(s[j]) {
j += 1
}
if j == i {
return "", fmt.Errorf("missing exponent part in %q", s)
}
return s[0 .. j]
}

fn scan_ident(s: &str) -> String {
    let i = 0;
    while i < s.len() {
        if is_ident_char(s[i]) {
            i += 1;
            continue
        }
        if s[i] != '\\' {
            break
        }
        i += 1;

        // Do not verify the next char, since it is escaped.
        // The next char may be encoded as multi-byte UTF8 sequence. See https://en.wikipedia.org/wiki/UTF-8#Encoding
        _, size = utf8.DecodeRuneInString(s[i .. ]);
        i += size
    }
    if i == 0 {
        panic("BUG: scan_ident couldn't find a single ident char; make sure is_ident_prefix called before scan_ident")
    }
    return s[0 .. i]
}

fn from_hex(ch: u8) -> i32 {
if ch >= '0' && ch <= '9' {
return int(ch - '0')
}
if ch >= 'a' && ch <= 'f' {
return int((ch - 'a') + 10)
}
if ch >= 'A' && ch <= 'F' {
return int((ch - 'A') + 10)
}
return -1
}

fn append_escaped_ident(dst []byte, s: &str) []byte {
for i = 0; i < s.len(); i += 1; {
ch = s[i]
if isIdentChar(ch) {
if i == 0 && !isFirstIdentChar(ch) {
// hex-encode the first char
dst.push('\\', 'x', toHex(ch>>4), toHex(ch&0xf))
} else {
dst.push(ch)
}
continue
}

// escape ch
dst.push('\\')
r, size = utf8.DecodeRuneInString(s[i .. ])
if r != utf8.RuneError && unicode.IsPrint(r) {
dst.push(s[i:i+size]...)
i += size - 1
} else {
// hex-encode non-printable chars
dst.push('x', toHex(ch>>4), toHex(ch&0xf))
}
}
return dst
}

fn is_eof(s: &str) -> bool {
return s.len() == 0
}

fn scan_tag_filter_op_prefix(s: &str) -> usize {
if s.len() >= 2 {
switch s[0 .. 2] {
case "=~", "!~", "!=":
return 2
}
}
if s.len() >= 1 {
if s[0] == '=' {
return 1
}
}
return -1
}

fn is_positive_number_prefix(s: &str) -> bool {
    if s.len() == 0 {
        return false
    }
    if is_decimal_char(s[0]) {
        return true
    }

    // Check for .234 numbers
    if s[0] != '.' || s.len() < 2 {
        return false
    }
    return is_decimal_char(s[1])
}

fn is_special_integer_prefix(s: &str) -> bool {
skipChars, _ = scanSpecialIntegerPrefix(s);
return skipChars > 0
}

fn is_positive_duration(s: &str) -> bool {
    let n = scan_duration(s);
    return n == s.len()
}

// duration_value returns the duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m, -2h5m or 2h-5m.
//
// The returned duration value can be negative.
fn duration_value(s: &str, step: i64) -> ParseResult<i64> {
if s.len() == 0 {
return 0, fmt.Errorf("duration cannot be empty")
}
// Try parsing floating-point duration
d = strconv.ParseFloat(s, 64);
if err == nil {
// Convert the duration to milliseconds.
return int64(d * 1000)
}
isMinus = false;
while s.len() > 0 {
    let n = scan_single_duration(s, true)?;
    ds = &s[0 .. n];
    s = &s[n .. ];
    dLocal = parse_single_duration(ds, step)?;
if err != nil {
return 0, err
}
if isMinus && dLocal > 0 {
dLocal = -dLocal
}
d += dLocal;
if dLocal < 0 {
isMinus = true
}
}
if math.Abs(d) > 1<<63-1 {
return 0, fmt.Errorf("too big duration %.0fms", d)
}
return int64(d)
}

// scan_duration scans duration, which must start with positive num.
//
// I.e. 123h, 3h5m or 3.4d-35.66s
fn scan_duration(s: &str) -> usize {
    // The first part must be non-negative
    let n = scan_single_duration(s, false);
    if n <= 0 {
        return -1
    }
    s = &s[n .. ];
    let mut i = n;
    loop {
        // Other parts may be negative
        n = scan_single_duration(s, true);
        if n <= 0 {
            return i
        }
        s = &s[n .. ];
        i += n
    }
}

fn scan_single_duration(s: &str, can_be_negative: bool) -> usize {
if s.len() == 0 {
return -1
}
let mut i = 0;
if s[0] == '-' && can_be_negative {
i += 1;
}
    while i < s.len() && isDecimalChar(s[i]) {
        i += 1;
    }
if i == 0 || i == s.len() {
return -1
}
    if s[i] == '.' {
        j = i;
        i += 1;
        while i < s.len() && isDecimalChar(s[i]) {
            i += 1;
        }
        if i == j || i == s.len() {
            return -1
        }
    }
switch s[i] {
case 'm':
if i+1 < s.len() && s[i+1] == 's' {
// duration in ms
return i + 2
}
// duration in minutes
return i + 1
case 's', 'h', 'd', 'w', 'y', 'i':
return i + 1
default:
return -1
}
}

fn is_decimal_char(ch: u8) -> bool {
return ch >= '0' && ch <= '9'
}

fn is_hex_char(ch: u8) -> bool {
return isDecimalChar(ch) || ch >= 'a' && ch <= 'f' || ch >= 'A' && ch <= 'F'
}

fn is_ident_prefix(s: &str) -> bool {
if s.len() == 0 {
return false
}
if s[0] == '\\' {
// Assume this is an escape char for the next char.
return true
}
return isFirstIdentChar(s[0])
}

fn is_first_ident_char(ch: u8) -> bool {
if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' {
return true
}
return ch == '_' || ch == ':'
}

fn is_ident_char(ch: u8) -> bool {
if isFirstIdentChar(ch) {
return true
}
return isDecimalChar(ch) || ch == '.'
}
