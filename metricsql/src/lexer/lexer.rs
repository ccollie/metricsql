use std::{cmp, fmt};
use crate::lexer::TokenKind;
use logos::{Logos, Span};
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

/// A byte-index tuple representing a span of characters in a string
///
/// Note that spans refer to the position in the input string as read by the
/// parser rather than the output of an expression's `Display` impl.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Default, Hash, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize
}

impl From<(usize, usize)> for TextSpan {
    fn from(tup: (usize, usize)) -> TextSpan {
        TextSpan::new(tup.0, tup.1)
    }
}

impl TextSpan {
    pub fn new(start: usize, end: usize) -> Self {
        TextSpan { start, end }
    }

    pub fn at(start: usize, len: usize) -> Self {
        Self {
            start,
            end: start + len - 1
        }
    }

    #[inline]
    pub fn cover(&self, other: TextSpan) -> TextSpan {
        let start = cmp::min(self.start, other.start);
        let end = cmp::max(self.end, other.end);
        TextSpan::new(start, end)
    }

    pub fn intersect_with(&mut self, other: TextSpan) -> bool {
        let start = cmp::max(self.start, other.start);
        let end = cmp::min(self.end, other.end);
        if end < start {
            return false;
        }
        self.start = start;
        self.end = end;
        true
    }
}

impl fmt::Display for TextSpan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.start, self.end)
    }
}


/// A token of metricsql source.
#[derive(Debug, Clone, PartialEq)]
pub struct Token<'source> {
    /// The kind of token.
    pub kind: TokenKind,
    pub text: &'source str,
    pub span: TextSpan,
}

impl<'source> Token<'source> {
    pub fn len(&self) -> usize {
        self.text.len()
    }
}

/// A lexer of metricsql source.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, TokenKind>,
    done: bool,
    peeked: VecDeque<Token<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(content: &'a str) -> Self {
        Self {
            inner: TokenKind::lexer(content),
            done: false,
            peeked: VecDeque::new(),
        }
    }

    pub fn is_eof(&self) -> bool {
        self.done
    }

    fn read_token(&mut self) -> Option<Token<'a>> {
        if self.done {
            return None;
        }

        if let Some(token) = self.peeked.pop_front() {
            return Some(token)
        }

        match self.inner.next() {
            None => {
                self.done = true;
                None
            }

            Some(kind) => {
                let Span { start, end } = self.inner.span();
                let span = TextSpan::new(start, end);
                Some(Token { kind, text: self.inner.slice(), span } )
            }
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    #[inline]
    fn next(&mut self) -> Option<Token<'a>> {
        self.read_token()
    }
}

#[cfg(test)]
mod tests {
    use super::{TokenKind::*, *};
    use test_case::test_case;
    use crate::lexer::{Lexer};

    macro_rules! test_tokens {
    ($src:expr, [$(
      $tok:expr
      $(=> $val:expr)?
    ),*$(,)?]) => {#[allow(unused)] {
      let src: &str = $src;
      let mut lex = Lexer::new(src);
      let mut index = 0;
      let mut offset = 0;
      $({
        let actual = lex.next().expect(&format!("Expected token {}", index + 1));
        let expected = $tok;
        assert_eq!(actual.kind, expected, "index: {}", index);
        $(
          assert_eq!(actual.text, $val, "index: {}", index);
        )?

        index += 1;
        offset += actual.len();
      })*

      match lex.next() {
        None => (),
        Some(t) => panic!("Expected exactly {} tokens, but got {:#?} when expecting EOF", index, t),
      }
    }};
  }

    #[test]
    fn empty() {
        test_tokens!("", []);
    }

    #[test]
    fn whitespace() {
        // whitespace is skipped
        test_tokens!("  \t\n\r\r\n", []);
    }

    #[test]
    fn negative_ints() {
        test_tokens!("-1", [Number]);
    }

    #[test_case("@", At)]
    #[test_case("{", LeftBrace)]
    #[test_case("}", RightBrace)]
    #[test_case("[", LeftBracket)]
    #[test_case("]", RightBracket)]
    #[test_case("(", LeftParen)]
    #[test_case(")", RightParen)]
    #[test_case(",", Comma)]
    #[test_case(";", SemiColon)]
    #[test_case(":", Colon)]
    #[test_case("=", Equal)]
    fn symbol(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("atan2", OpAtan2)]
    #[test_case("and", OpAnd)]
    #[test_case("or", OpOr)]
    #[test_case("if", OpIf)]
    #[test_case("ifnot", OpIfNot)]
    #[test_case("unless", OpUnless)]
    #[test_case("==", OpEqual)]
    #[test_case("!=", OpNotEqual)]
    #[test_case("=~", RegexEqual)]
    #[test_case("!~", RegexNotEqual)]
    #[test_case("+", OpPlus)]
    #[test_case("-", OpMinus)]
    #[test_case("*", OpMul)]
    #[test_case("/", OpDiv)]
    #[test_case("%", OpMod)]
    #[test_case("^", OpPow)]
    #[test_case("<", OpLessThan)]
    #[test_case(">", OpGreaterThan)]
    #[test_case("<=", OpLessThanOrEqual)]
    #[test_case(">=", OpGreaterThanOrEqual)]
    fn operator(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("1")]
    #[test_case("73")]
    #[test_case("65535")]
    fn number_int(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("1.0")]
    #[test_case("0.10")]
    #[test_case("0e100")]
    #[test_case("1e100")]
    #[test_case("1.1e100")]
    #[test_case("1.2e-100")]
    #[test_case("1.3e+100")]
    #[test_case("0.49641")]
    fn number_float(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("0b1110")]
    #[test_case("0b0")]
    fn number_binary(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("0x1fffa")]
    #[test_case("0x4")]
    fn number_hex(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test_case("03")]
    #[test_case("0o1")]
    #[test_case("0012")]
    fn number_octal(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test]
    fn number_inf() {
        test_tokens!("inf", [Number]);
        test_tokens!("InF", [Number]);
        test_tokens!("INF", [Number]);
    }

    #[test]
    fn number_nan() {
        test_tokens!("nan", [Number]);
        test_tokens!("nAN", [Number]);
        test_tokens!("NAN", [Number]);
    }

    #[test]
    fn misc_numbers() {
        // Numbers
        let s = r"3+1.2-.23+4.5e5-78e-6+1.24e+45-NaN+Inf";
        let expected = vec!["3", "+", "1.2", "-", ".23", "+", "4.5e5", "-", "78e-6", "+", "1.24e+45", "-", "NaN", "+", "Inf"];
        test_success(s, &expected);

        let s = "12.34 * 0X34 + 0b11 + 0O77";
        let expected = vec!["12.34", "*", "0X34", "+", "0b11", "+", "0O77"];
        test_success(s, &expected);
    }

    #[test_case("1.+", ErrorNumJunkAfterDecimalPoint)]
    #[test_case("1e!", ErrorNumJunkAfterExponent)]
    #[test_case("1e+!", ErrorNumJunkAfterExponentSign)]
    fn bad_number(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("\"hi\"", LiteralString; "double_1")]
    #[test_case("\"hi\n\"", LiteralString; "double_2")]
    #[test_case("\"hi\\\"\"", LiteralString; "double_3")]
    #[test_case("'hi'", LiteralString; "single_1")]
    #[test_case("'hi\n'", LiteralString; "single_2")]
    #[test_case("'hi\\''", LiteralString; "single_3")]
    fn string(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("\"hi", ErrorStringDoubleQuotedUnterminated; "double_unterm")]
    #[test_case("'hi", ErrorStringSingleQuotedUnterminated; "single_unterm")]
    fn string_unterminated(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("by", By)]
    #[test_case("bool", Bool)]
    #[test_case("default", OpDefault)]
    #[test_case("group_left", GroupLeft)]
    #[test_case("group_right", GroupRight)]
    #[test_case("ignoring", Ignoring)]
    #[test_case("keep_metric_names", KeepMetricNames)]
    #[test_case("limit", Limit)]
    #[test_case("offset", Offset)]
    #[test_case("on", On)]
    #[test_case("with", With)]
    #[test_case("without", Without)]
    fn keyword(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test]
    fn identifier() {
        test_tokens!("foobar123", [Ident=>"foobar123"]);
    }

    #[test_case("123ms")]
    #[test_case("123s")]
    #[test_case("123m")]
    #[test_case("1h")]
    #[test_case("2d")]
    #[test_case("3w")]
    #[test_case("4y")]
    #[test_case("1i")]
    #[test_case("3i")]
    fn integer_duration(src: &str) {
        test_tokens!(src, [Duration]);
    }

    #[test_case("123.45ms")]
    #[test_case("0.234s")]
    #[test_case("1.5s")]
    #[test_case("1.5h")]
    #[test_case("1.2h")]
    #[test_case("1.1d")]
    #[test_case("2.3w")]
    #[test_case("1.4y")]
    #[test_case("0.1i")]
    #[test_case("1.5m3.4s2.4ms")]
    #[test_case("-1.5m3.4s2.9ms")]
    fn float_duration(src: &str) {
        test_tokens!(src, [Duration]);
    }

    #[test]
    fn various_durations() {
        // Various durations
        test_success("m offset 123h", &["m", "offset", "123h"]);

        let mut s = "m offset -1.23w-5h34.5m - 123";
        let expected = vec!["m", "offset", "-", "1.23w-5h34.5m", "-", "123"];
        test_success(s, &expected);
    }

    #[test_case("2k")]
    #[test_case("2.3kb")]
    #[test_case("3ki")]
    #[test_case("4.5kib")]
    #[test_case("2M")]
    #[test_case("2.3MB")]
    #[test_case("3mi")]
    #[test_case("4.5Mib")]
    #[test_case("2G")]
    #[test_case("2.3gB")]
    #[test_case("3Gi")]
    #[test_case("4.5GiB")]
    #[test_case("2T")]
    #[test_case("2.3tb")]
    #[test_case("3ti")]
    #[test_case("-4.5TIB")]
    fn number_with_unit(src: &str) {
        test_tokens!(src, [NumberWithUnit]);
    }

    #[test]
    fn identifiers() {
        test_tokens!(
          "foo bar123",
          [
            Ident=>"foo",
            Ident=>"bar123",
          ]
        );
    }

    #[test]
    fn py_comment() {
        // comments are skipped
        test_tokens!("# hi", []);
    }

    #[test]
    fn junk() {
        let src = "💩";
        test_tokens!(src, [ErrorInvalidToken]);
    }

    #[test]
    fn metric_name() {
        // Just metric name
        test_success("metric", &["metric"]);

        // Metric name with spec chars
        test_success("foo.bar_", &["foo.bar_"]);
    }

    #[test]
    fn metric_name_with_window() {
        // Metric name with window
        let s = "metric[5m]  ";
        test_success(s, &["metric", "[", "5m", "]"]);
    }

    #[test]
    fn metric_name_with_offset() {
        // Metric name with offset
        let s = "   metric offset 10d   ";
        test_success(s, &["metric", "offset", "10d"]);
    }

    #[test]
    fn metric_name_with_tag_filters() {
        // Metric name with tag filters
        let s = r#"  metric:12.34{a="foo", b != "bar", c=~ "x.+y", d !~ "zzz"}"#;
        let expected = vec!["metric:12.34", "{", "a", "=", r#""foo""#, ",", "b",
                        "!=", r#""bar""#, ",", "c", "=~", r#""x.+y""#, ",", "d", "!~", "zzz", "}"];

        test_success(s, &expected);
    }

    #[test]
    fn function_call() {
        // fn call
        let s = r#"sum  (  metric{x="y"  }  [5m] offset 10h)"#;
        let expected = vec!["sum", "(", "metric", "{", "x", "=", "y", "}", "[", "5m", "]", "offset", "10h", ")"];
        test_success(s, &expected);
    }

    #[test]
    fn binary_op() {
        // Binary op
        let s = "a+b or c % d and e unless f";
        let expected = vec!["a", "+", "b", "or", "c", "%", "d", "and", "e", "unless", "f"];
        test_success(s, &expected);
    }

    #[test]
    fn comments() {
        let s = r"# comment # sdf
		foobar # comment
		baz
		# yet another comment";
        test_success(s, &["foobar", "baz"])
    }

    #[test]
    fn strings() {
        // An empty string
        test_success("", &[]);

        // String with whitespace
        let mut s = "  \n\t\r ";
        test_success(s, &[]);

        // Strings
        let s = r#""''"#.to_owned() + "``" + r#"\\"  '\\'  "\"" '\''"\\\"\\""#;
        let expected = vec![r#""""#, "''", "``", r#"\\"", `'\\'", `"\""", `'\''" "\\\"\\"#];
        test_success(&s, &expected);
    }

    #[test]
    fn miscellaneeous() {
        let s = "   `foo\\\\\\`бар`  ";
        let expected = vec!["`foo\\\\\\`бар`"];
        test_success(s, &expected);
    }

    fn test_success(s: &str, expected_tokens: &[&str]) {
        let lex = Lexer::new(s);

        let mut i = 0;
        for tok in lex {
            assert_eq!(tok.text, expected_tokens[i]);
            i += 1;
        }
    }

}
