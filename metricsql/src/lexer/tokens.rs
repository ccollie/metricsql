use logos::Logos;
use std::fmt::{Display, Formatter};

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
#[logos(subpattern decimal = r"[0-9][_0-9]*")]
#[logos(subpattern hex = r"-?0[xX][0-9a-fA-F][_0-9a-fA-F]*")]
#[logos(subpattern octal = r"0[oO](_?[0-7]+)+")]
#[logos(subpattern binary = r"0[bB][0-1][_0-1]*")]
#[logos(subpattern float = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")]
#[logos(subpattern exp = r"[eE][+-]?[0-9][_0-9]*")]
#[logos(subpattern duration = r"(-?)(0|[1-9]\d*)(\.\d+)?(ms|s|m|h|d|w|y|i)+")]
pub enum TokenKind {
    #[token("and", ignore(ascii_case))]
    OpAnd,

    #[token("atan2", ignore(ascii_case))]
    OpAtan2,

    #[token("bool", ignore(ascii_case))]
    Bool,

    #[token("by", ignore(ascii_case))]
    By,

    #[token("default", ignore(ascii_case))]
    OpDefault,

    #[token("group_left", ignore(ascii_case))]
    GroupLeft,

    #[token("group_right", ignore(ascii_case))]
    GroupRight,

    #[token("if", ignore(ascii_case))]
    OpIf,

    #[token("ifNot", ignore(ascii_case))]
    OpIfNot,

    #[token("ignoring", ignore(ascii_case))]
    Ignoring,

    #[token("keep_metric_names", ignore(ascii_case))]
    KeepMetricNames,

    #[token("limit", ignore(ascii_case))]
    Limit,

    #[token("on", ignore(ascii_case))]
    On,

    #[token("offset", ignore(ascii_case))]
    Offset,

    #[token("or", ignore(ascii_case))]
    OpOr,

    #[token("unless", ignore(ascii_case))]
    OpUnless,

    #[token("with", ignore(ascii_case))]
    With,

    #[token("without", ignore(ascii_case))]
    Without,

    #[regex("(?&duration)+", priority = 5)]
    Duration,

    #[token("nan", ignore(ascii_case))]
    #[token("inf", ignore(ascii_case))]
    #[regex("(?&hex)")]
    #[regex("(?&octal)")]
    #[regex("(?&binary)")]
    #[regex("(?&float)")]
    Number,

    #[regex(r"[_a-zA-Z][_a-zA-Z0-9:\.]*")]
    Ident,

    #[regex("(?&float)(?i)(kib|ki|kb|k|mib|mi|mb|m|gib|gi|gb|g|tib|ti|tb|t)", priority = 8)]
    NumberWithUnit,

    #[regex(r"(?:0|[1-9][0-9]*)\.[^0-9]")]
    ErrorNumJunkAfterDecimalPoint,

    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][^+\-0-9]")]
    ErrorNumJunkAfterExponent,

    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?[eE][+-][^0-9]")]
    ErrorNumJunkAfterExponentSign,

    #[token("@")]
    At,

    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token(";")]
    SemiColon,

    #[token("=")]
    Equal,

    #[token("==")]
    OpEqual,

    #[token("!=")]
    OpNotEqual,

    #[token("<")]
    OpLessThan,

    #[token("<=")]
    OpLessThanOrEqual,

    #[token(">")]
    OpGreaterThan,

    #[token(">=")]
    OpGreaterThanOrEqual,

    #[token("+")]
    OpPlus,

    #[token("-")]
    OpMinus,

    #[token("/")]
    OpDiv,

    #[token("*")]
    OpMul,

    #[token("^")]
    OpPow,

    #[token("%")]
    OpMod,

    #[token("=~")]
    RegexEqual,

    #[token("!~")]
    RegexNotEqual,

    #[regex("'(?s:[^'\\\\]|\\\\.)*'")]
    #[regex("`(?s:[^`\\\\]|\\\\.)*`")]
    #[regex("\"(?s:[^\"\\\\]|\\\\.)*\"")]
    StringLiteral,

    #[regex("\"(?s:[^\"\\\\]|\\\\.)*")]
    ErrorStringDoubleQuotedUnterminated,

    #[regex("'(?s:[^'\\\\]|\\\\.)*")]
    ErrorStringSingleQuotedUnterminated,

    #[regex("@[^\"'\\s]\\S+")]
    ErrorStringMissingQuotes,

    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,

    #[regex(r"#[^\r\n]*(\r\n|\n)?", logos::skip)]
    SingleLineHashComment,

    Eof,

    #[error]
    ErrorInvalidToken,
}

impl TokenKind {
    pub fn is_trivia(&self) -> bool {
        matches!(self, Self::Whitespace | Self::SingleLineHashComment)
    }

    pub fn is_error(&self) -> bool {
        use TokenKind::*;

        matches!(self,
            ErrorStringDoubleQuotedUnterminated
            | ErrorStringSingleQuotedUnterminated
            | ErrorNumJunkAfterDecimalPoint
            | ErrorNumJunkAfterExponent
            | ErrorNumJunkAfterExponentSign
            | ErrorStringMissingQuotes
            | ErrorInvalidToken)
    }

    pub fn is_operator(&self) -> bool {
        use TokenKind::*;

        matches!(self, OpAtan2 | OpMul | OpDiv | OpDefault | OpIf | OpIfNot | OpMod | OpPlus | OpMinus | OpLessThan
            | OpGreaterThan | OpLessThanOrEqual | OpGreaterThanOrEqual | OpEqual | OpNotEqual | OpPow
            | OpAnd | OpOr | OpUnless)
    }

    #[inline]
    pub fn is_comparison_op(&self) -> bool {
        use TokenKind::*;
        matches!(self, OpEqual | OpNotEqual | OpGreaterThanOrEqual | OpGreaterThan | OpLessThanOrEqual | OpLessThan)
    }

    #[inline]
    pub fn is_rollup_start(&self) -> bool {
        use TokenKind::*;

        matches!(self, Offset | At | LeftBracket)
    }

    #[inline]
    pub fn is_group_modifier(&self) -> bool {
        use TokenKind::*;
        matches!(self, On | Ignoring)
    }

    #[inline]
    pub fn is_join_modifier(&self) -> bool {
        use TokenKind::*;
        matches!(self, GroupLeft | GroupRight)
    }

    pub fn is_aggregate_modifier(&self) -> bool {
        use TokenKind::*;
        matches!(self, By | Without)
    }

    /// tokens that can function as identifiers in certain constructions (functions/metric names)
    /// not a good idea, but we're matching the original
    pub fn is_ident_like(&self) -> bool {
        use TokenKind::*;
        // keywords
        matches!(self, By | Bool | OpDefault | GroupLeft | GroupRight |
            Ignoring | KeepMetricNames | Limit | On | Offset | With | Without |
            OpAnd | OpAtan2 | OpIf | OpIfNot | OpOr | OpUnless)
    }

    pub fn is_error_token(&self) -> bool {
        use TokenKind::*;
        matches!(self,
            ErrorInvalidToken
            | ErrorNumJunkAfterDecimalPoint
            | ErrorStringMissingQuotes
            | ErrorNumJunkAfterExponent
            | ErrorNumJunkAfterExponentSign
            | ErrorStringDoubleQuotedUnterminated
            | ErrorStringSingleQuotedUnterminated)
    }
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            // keywords
            Self::By => "by",
            Self::Bool => "bool",
            Self::OpDefault => "default",
            Self::GroupLeft => "group_left",
            Self::GroupRight => "group_right",
            Self::Ignoring => "ignoring",
            Self::KeepMetricNames => "keep_metric_names",
            Self::Limit => "limit",
            Self::On => "on",
            Self::Offset => "offset",
            Self::With => "with",
            Self::Without => "without",
            // variable tokens
            Self::Duration => "<duration>",
            Self::Ident => "<ident>",
            Self::Number => "<number>",
            Self::NumberWithUnit => "<number><unit>",
            // symbols
            Self::At => "@",
            Self::LeftBrace => "{{",
            Self::RightBrace => "}}",
            Self::LeftBracket => "[",
            Self::RightBracket => "]",
            Self::Colon => ":",
            Self::Comma => ",",
            Self::LeftParen => "(",
            Self::RightParen => ")",
            Self::SemiColon => ";",
            Self::Equal => "=",
            // operators
            Self::OpAnd => "and",
            Self::OpAtan2 => "atan2",
            Self::OpIf => "if",
            Self::OpIfNot => "ifnot",
            Self::OpOr => "or",
            Self::OpUnless => "unless",
            Self::OpMul => "*",
            Self::OpDiv => "/",
            Self::OpMod => "%",
            Self::OpPlus => "+",
            Self::OpMinus => "-",
            Self::OpLessThan => "<",
            Self::OpGreaterThan => ">",
            Self::OpLessThanOrEqual => "<=",
            Self::OpGreaterThanOrEqual => ">=",
            Self::OpEqual => "==",
            Self::OpNotEqual => "!=",
            Self::OpPow => "^",
            Self::RegexEqual => "=~",
            Self::RegexNotEqual => "!~",
            // strings
            Self::StringLiteral => "<string>",
            Self::Whitespace => "<ws>",
            Self::SingleLineHashComment => "<#comment>",

            // string errors
            Self::ErrorStringDoubleQuotedUnterminated => "<unterminated double quoted string>",
            Self::ErrorStringSingleQuotedUnterminated => "<unterminated single quoted string>",
            Self::ErrorStringMissingQuotes => "<string missing quotes>",

            // number errors
            Self::ErrorNumJunkAfterDecimalPoint => "invalid number (unexpected character after decimal point)",
            Self::ErrorNumJunkAfterExponent => "invalid number (unexpected character after exponent)",
            Self::ErrorNumJunkAfterExponentSign => "invalid number (unexpected character after exponent sign)",

            // other
            Self::ErrorInvalidToken => "<invalid token>",
            Self::Eof => "<Eof>"
        })
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
    #[test_case("ifNot", OpIfNot)]
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
    #[test_case("0O12")]
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

    #[test_case("\"hi\"", StringLiteral; "double_1")]
    #[test_case("\"hi\n\"", StringLiteral; "double_2")]
    #[test_case("\"hi\\\"\"", StringLiteral; "double_3")]
    #[test_case("'hi'", StringLiteral; "single_1")]
    #[test_case("'hi\n'", StringLiteral; "single_2")]
    #[test_case("'hi\\''", StringLiteral; "single_3")]
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

        let s = "m offset -1.23w-5h34.5m - 123";
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
        let s = "  \n\t\r ";
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
