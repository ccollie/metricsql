

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

fn (lex *lexer) Context() string {
return fmt.Sprintf("%s%s", lex.Token, lex.sTail)
}

fn (lex *lexer) Init(s: &str) {
lex.Token = ""
lex.prevTokens = nil
lex.nextTokens = nil
lex.err = nil

lex.sOrig = s
lex.sTail = s
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

fn (lex *lexer) next() (string, error) {
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

fn scan_string(s: &str) (string, error) {
if s.len() < 2 {
return "", fmt.Errorf("cannot find end of string in %q", s)
}

quote = s[0]
i = 1
for {
n = strings.IndexByte(s[i .. ], quote)
if n < 0 {
return "", fmt.Errorf("cannot find closing quote %c for the string %q", quote, s)
}
i += n
bs = 0
for bs < i && s[i-bs-1] == '\\' {
bs++
}
if bs%2 == 0 {
token = s[0 .. i+1]
return token
}
i += 1;
}
}

fn parse_positive_number(s: &str) -> ParseResult<f64> {
    if is_special_integer_prefix(s) {
        n = strconv.ParseInt(s, 0, 64);
        if err != nil {
            return 0, err
        }
return float64(n)
}
s = s.to_ascii_lowercase();
m = float64(1);
switch true {
case s.ends_with("kib"):
s = &s[0 .. .len()-3];
m = 1024;
case s.ends_with("ki"):
s = &s[0 .. .len()-2];
m = 1024;
case s.ends_with("kb"):
s = &s[0 .. .len()-2];
m = 1000;
case s.ends_with("k"):
s = &s[0 .. .len()-1];
m = 1000;
case s.ends_with("mib"):
s = &s[0 ..len()-3];
m = 1024 * 1024;
case s.ends_with("mi"):
s = &s[0 ..len()-2];
m = 1024 * 1024;
case s.ends_with("mb"):
s = &s[0 ..len()-2];
m = 1000 * 1000;
case s.ends_with("m"):
s = &s[0 ..len()-1];
m = 1000 * 1000;
case s.ends_with("gib"):
s = &s[0 ..len()-3];
m = 1024 * 1024 * 1024;
case s.ends_with("gi"):
s = &s[0 ..len()-2];
m = 1024 * 1024 * 1024;
case s.ends_with("gb"):
s = &s[0 ..len()-2];
m = 1000 * 1000 * 1000;
case s.ends_with("g"):
s = &s[0 ..len()-1];
m = 1000 * 1000 * 1000;
case s.ends_with("tib"):
s = &s[0 ..len()-3];
m = 1024 * 1024 * 1024 * 1024;
case s.ends_with("ti"):
s = &s[0 ..len()-2];
m = 1024 * 1024 * 1024 * 1024;
case s.ends_with("tb"):
s = &s[0 ..len()-2];
m = 1000 * 1000 * 1000 * 1000;
case s.ends_with("t"):
s = &s[0 ..len()-1];
m = 1000 * 1000 * 1000 * 1000
}
v = strconv.ParseFloat(s, 64);
if err != nil {
return 0, err
}
return v * m
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

fn scan_num_multiplier(s: &str) int {
s = s.to_ascii_lowercase()
switch true {
case strings.HasPrefix(s, "kib"):
return 3
case strings.HasPrefix(s, "ki"):
return 2
case strings.HasPrefix(s, "kb"):
return 2
case strings.HasPrefix(s, "k"):
return 1
case strings.HasPrefix(s, "mib"):
return 3
case strings.HasPrefix(s, "mi"):
return 2
case strings.HasPrefix(s, "mb"):
return 2
case strings.HasPrefix(s, "m"):
return 1
case strings.HasPrefix(s, "gib"):
return 3
case strings.HasPrefix(s, "gi"):
return 2
case strings.HasPrefix(s, "gb"):
return 2
case strings.HasPrefix(s, "g"):
return 1
case strings.HasPrefix(s, "tib"):
return 3
case strings.HasPrefix(s, "ti"):
return 2
case strings.HasPrefix(s, "tb"):
return 2
case strings.HasPrefix(s, "t"):
return 1
default:
return 0
}
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
        _, size = utf8.DecodeRuneInString(s[i .. ])
        i += size
    }
    if i == 0 {
        panic("BUG: scan_ident couldn't find a single ident char; make sure is_ident_prefix called before scan_ident")
    }
    return s[0 .. i]
}

fn unescape_ident(s: &str) -> String {
    n = strings.IndexByte(s, '\\');
if n < 0 {
return s
}
dst = make([]byte, 0, s.len());
for {
    dst.push(&s[0 .. n];...)
    s = &s[n+1 .. ];
    if s.len() == 0 {
        return string(dst)
    }
    if s[0] == 'x' && s.len() >= 3 {
        h1 = from_hex(s[1]);
        h2 = from_hex(s[2]);
        if h1 >= 0 && h2 >= 0 {
            dst.push(byte((h1<<4)|h2));
            s = s[3 .. ];
        } else {
dst.push(s[0]);
s = s[1 .. ]
}
} else {
// UTF8 char. See https://en.wikipedia.org/wiki/UTF-8#Encoding
_, size = utf8.DecodeRuneInString(s);
dst.push(&s[0 .. ize]...)
s = s[size .. ]
}
n = strings.IndexByte(s, '\\');
if n < 0 {
dst.push(s...)
return string(dst)
}
}
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

fn toHex(n: u8) -> char {
if n < 10 {
return '0' + n
}
return 'a' + (n - 10)
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

fn (lex *lexer) Prev() {
lex.nextTokens = append(lex.nextTokens, lex.Token)
lex.Token = lex.prevTokens[len(lex.prevTokens)-1]
lex.prevTokens = lex.prevTokens[0 .. len(lex.prevTokens)-1]
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

fn isInfOrNaN(s: &str) -> bool {
if s.len() != 3 {
return false
}
s = s.to_ascii_lowercase();
return s == "inf" || s == "nan"
}

fn is_offset(s: &str) -> bool {
s = s.to_ascii_lowercase();
return s == "offset"
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

fn scan_special_integer_prefix(s: &str) (skipChars int, isHex bool) {
if s.len() < 1 || s[0] != '0' {
return 0, false
}
s = strings.ToLower(s[1 .. ])
if s.len() == 0 {
return 0, false
}
if isDecimalChar(s[0]) {
// octal number: 0123
return 1, false
}
if s[0] == 'x' {
// 0x
return 2, true
}
if s[0] == 'o' || s[0] == 'b' {
// 0x, 0o or 0b prefix
return 2, false
}
return 0, false
}

fn is_positive_duration(s: &str) -> bool {
    let n = scan_duration(s);
    return n == s.len()
}

// positive_duration_value returns positive duration in milliseconds for the given s
// and the given step.
//
// Duration in s may be combined, i.e. 2h5m or 2h-5m.
//
// Error is returned if the duration in s is negative.
fn positive_duration_value(s: &str, step: i64) -> ParseResult<i64> {
    let d = duration_value(s, step);
    if d < 0 {
        return 0, fmt.Errorf("duration cannot be negative; got %q", s)
    }
    return d
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

fn parse_single_duration(s: &str, step: i64) -> ParseResult<f64> {
numPart = &s[0..s.len()-1];
if strings.HasSuffix(numPart, "m") {
// Duration in ms
numPart = numPart[0 .. num_part.len()-1]
}
f = strconv.ParseFloat(numPart, 64);
if err != nil {
return 0, fmt.Errorf("cannot parse duration %q: %s", s, err)
}
var mp float64;
switch s[num_part.len() .. ] {
case "ms":
mp = 1e-3;
case "s":
mp = 1
case "m":
mp = 60;
case "h":
mp = 60 * 60;
case "d":
mp = 24 * 60 * 60;
case "w":
mp = 7 * 24 * 60 * 60;
case "y":
mp = 365 * 24 * 60 * 60;
case "i":
mp = float64(step) / 1e3;
default:
return 0, fmt.Errorf("invalid duration suffix in %q", s)
}
return mp * f * 1e3
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

fn is_space_char(ch: u8) -> bool {
switch ch {
case ' ', '\t', '\n', '\v', '\f', '\r':
return true
default:
return false
}
}