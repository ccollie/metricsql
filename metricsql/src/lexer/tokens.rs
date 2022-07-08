use logos::Logos;
use std::fmt::{Display, Formatter};

static DURATION_REGEX: &str = r"(-?)(?=\d)(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)([DW]))?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?";

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
pub enum TokenKind {
    #[token("and", ignore(ascii_case))]
    And,

    #[token("atan2", ignore(ascii_case))]
    Atan2,

    #[token("bool", ignore(ascii_case))]
    Bool,

    #[token("by", ignore(ascii_case))]
    By,

    #[token("default", ignore(ascii_case))]
    Default,

    #[token("group_left", ignore(ascii_case))]
    GroupLeft,

    #[token("group_right", ignore(ascii_case))]
    GroupRight,

    #[token("if", ignore(ascii_case))]
    If,

    #[token("ifnot", ignore(ascii_case))]
    IfNot,

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
    Or,

    #[token("unless", ignore(ascii_case))]
    Unless,

    #[token("with", ignore(ascii_case))]
    With,

    #[token("without", ignore(ascii_case))]
    Without,

    #[regex(r"(-?)(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)([DW]))?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", priority = 20)]
    Duration,

    #[regex(r"[_a-zA-Z][_a-zA-Z0-9]*")]
    Ident,

    #[regex("[-|+]?[i|I][n|N][f|F]")]
    #[token("[-|+]?[n|N][a|A][n|N]")]
    #[regex("0x[0-9a-fA-F]+")]
    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")]
    Number,

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

    #[token(".")]
    Dot,

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
    NotEqual,

    #[token("<")]
    LessThan,

    #[token("<=")]
    LessThanOrEqual,

    #[token(">")]
    GreaterThan,

    #[token(">=")]
    GreaterThanOrEqual,

    #[token("+")]
    OpPlus,

    #[token("-")]
    OpMinus,

    #[token("/")]
    OpDiv,

    #[token("*")]
    OpMul,

    #[token("^")]
    Pow,

    #[token("%")]
    OpMod,

    #[token("=~")]
    RegexEqual,

    #[token("!~")]
    RegexNotEqual,

    #[regex("'(?s:[^'\\\\]|\\\\.)*'")]
    #[regex("\"(?s:[^\"\\\\]|\\\\.)*\"")]
    QuotedString,

    #[regex("\"(?s:[^\"\\\\]|\\\\.)*")]
    ErrorStringDoubleQuotedUnterminated,

    #[regex("'(?s:[^'\\\\]|\\\\.)*")]
    ErrorStringSingleQuotedUnterminated,

    #[regex("@[^\"'\\s]\\S+")]
    ErrorStringMissingQuotes,

    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,

    #[regex(r"[^\r\n]*(\r\n|\n)?", logos::skip)]
    SingleLineHashComment,

    #[error]
    Error,
}

impl TokenKind {
    pub fn is_trivia(&self) -> bool {
        matches!(self, Self::Whitespace | Self::SingleLineHashComment)
    }

    pub fn is_error(&self) -> bool {
        use TokenKind::*;

        matches!(self, ErrorStringDoubleQuotedUnterminated
            | ErrorStringSingleQuotedUnterminated
            | ErrorNumJunkAfterDecimalPoint
            | ErrorNumJunkAfterExponent
            | ErrorNumJunkAfterExponentSign
            | ErrorStringMissingQuotes
            | Error)
    }

    pub fn is_operator(&self) -> bool {
        use TokenKind::*;

        matches!(self, Atan2 | OpMul | OpDiv | If | IfNot | OpMod | OpPlus | OpMinus | LessThan
            | GreaterThan | LessThanOrEqual | GreaterThanOrEqual | OpEqual | NotEqual | Pow
            | And | Or | Unless)
    }

    #[inline]
    pub fn is_comparison_op(&self) -> bool {
        use TokenKind::*;

        match self {
            OpEqual | NotEqual | GreaterThanOrEqual | GreaterThan | LessThanOrEqual | LessThan => {
                true
            }
            _ => false,
        }
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

    pub fn is_error_token(&self) -> bool {
        use TokenKind::*;
        matches!(self, ErrorNumJunkAfterDecimalPoint
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
            Self::And => "and",
            Self::By => "by",
            Self::Bool => "bool",
            Self::Default => "default",
            Self::GroupLeft => "group_left",
            Self::GroupRight => "group_right",
            Self::If => "if",
            Self::IfNot => "ifnot",
            Self::Ignoring => "ignoring",
            Self::KeepMetricNames => "keep_metric_names",
            Self::Limit => "limit",
            Self::On => "on",
            Self::Offset => "offset",
            Self::Or => "or",
            Self::Unless => "unless",
            Self::With => "with",
            Self::Without => "without",
            // variable tokens
            Self::Duration => "<duration>",
            Self::Ident => "<ident>",
            Self::Number => "<number>",
            // symbols
            Self::At => "@",
            Self::LeftBrace => "{{",
            Self::RightBrace => "}}",
            Self::LeftBracket => "[",
            Self::RightBracket => "]",
            Self::Colon => ":",
            Self::Comma => ",",
            Self::Dot => ".",
            Self::LeftParen => "(",
            Self::RightParen => ")",
            Self::SemiColon => ";",
            Self::Equal => "=",
            // operators
            Self::Atan2 => "atan2",
            Self::OpMul => "*",
            Self::OpDiv => "/",
            Self::OpMod => "%",
            Self::OpPlus => "+",
            Self::OpMinus => "-",
            Self::LessThan => "<",
            Self::GreaterThan => ">",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThanOrEqual => ">=",
            Self::OpEqual => "==",
            Self::NotEqual => "!=",
            Self::Pow => "^",
            Self::RegexEqual => "=~",
            Self::RegexNotEqual => "!~",
            // strings
            Self::QuotedString => "<string>",
            Self::Whitespace => "<ws>",
            Self::SingleLineHashComment => "<#comment>",

            // string errors
            Self::ErrorStringDoubleQuotedUnterminated => "<unterminated double quoted string>",
            Self::ErrorStringSingleQuotedUnterminated => "<unterminated single quoted string>",
            Self::ErrorStringMissingQuotes => "<string missing quotes>",

            // number errors
            Self::ErrorNumJunkAfterDecimalPoint => "<unexpected character after decimal point>",
            Self::ErrorNumJunkAfterExponent => "<unexpected character after exponent>",
            Self::ErrorNumJunkAfterExponentSign => "<unexpected character after exponent sign>",

            // other
            Self::Error => "<invalid token>",
        })
    }
}
#[cfg(test)]
mod tests {
    use super::{TokenKind::*, *};
    use test_case::test_case;

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
          let val = &src[offset as usize..(offset + actual.len) as usize];
          assert_eq!(val, $val, "index: {}", index);
        )?

        index += 1;
        offset += actual.len;
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
        test_tokens!("  \t\n\r\r\n", [Whitespace]);
    }

    #[test_case("@", At)]
    #[test_case("{", LeftBrace)]
    #[test_case("}", RightBrace)]
    #[test_case("[", LeftBracket)]
    #[test_case("]", RightBracket)]
    #[test_case("(", LeftParen)]
    #[test_case(")", lRightParen)]
    #[test_case(",", Comma)]
    #[test_case(".", Dot)]
    #[test_case(";", SemiColon)]
    #[test_case("=", Equal)]
    fn symbol(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case(":", OpColon)]
    #[test_case("atan2", OpAtan2)]
    #[test_case("==", OpEqual)]
    #[test_case("!=", OpNotEqual)]
    #[test_case("=~", OpRegexEqual)]
    #[test_case("!~", OpRegexNotEqual)]
    #[test_case("+", OpPlus)]
    #[test_case("-", OpMinus)]
    #[test_case("*", OpMul)]
    #[test_case("/", OpDiv)]
    #[test_case("%", OpMod)]
    #[test_case("&", OpBitAnd)]
    #[test_case("^", OpPow)]
    #[test_case("<", OpLessThan)]
    #[test_case(">", OpGreaterThan)]
    #[test_case("<=", OpLessThanOrEqual)]
    #[test_case(">=", OpGreaterThanOrEqual)]
    fn operator(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("->"; "arrow_right")]
    #[test_case("<-"; "arrow_left")]
    #[test_case(">==|"; "junk")]
    fn bad_op(src: &str) {
        test_tokens!(src, [ErrorUnknownOperator]);
    }

    #[test_case("1")]
    #[test_case("1.0")]
    #[test_case("0.10")]
    #[test_case("0e100")]
    #[test_case("1e100")]
    #[test_case("1.1e100")]
    #[test_case("1.2e-100")]
    #[test_case("1.3e+100")]
    fn number(src: &str) {
        test_tokens!(src, [Number]);
    }

    #[test]
    fn number_0100() {
        test_tokens!("0100", [Number=>"0", Number=>"100",]);
    }

    #[test]
    fn number_10_p_10() {
        test_tokens!(
          "10+11",
          [
            Number=>"10",
            OpPlus,
            Number=>"11",
          ]
        );
    }

    #[test_case("1.+", ErrorNumJunkAfterDecimalPoint)]
    #[test_case("1e!", ErrorNumJunkAfterExponent)]
    #[test_case("1e+!", ErrorNumJunkAfterExponentSign)]
    fn bad_number(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("\"hi\"", StringDoubleQuoted; "double_1")]
    #[test_case("\"hi\n\"", StringDoubleQuoted; "double_2")]
    #[test_case("\"hi\\\"\"", StringDoubleQuoted; "double_3")]
    #[test_case("'hi'", StringSingleQuoted; "single_1")]
    #[test_case("'hi\n'", StringSingleQuoted; "single_2")]
    #[test_case("'hi\\''", StringSingleQuoted; "single_3")]
    fn string(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("\"hi", ErrorStringDoubleQuotedUnterminated; "double_unterm")]
    #[test_case("'hi", ErrorStringSingleQuotedUnterminated; "single_unterm")]
    fn string_unterminated(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test_case("atan2", OpAtan2)]
    #[test_case("by", By)]
    #[test_case("default", Default)]
    #[test_case("group_left", GroupLeft)]
    #[test_case("group_right", GroupRight)]
    #[test_case("if", OpIf)]
    #[test_case("ifnot", OpIfNot)]
    #[test_case("ignoring", Ignoring)]
    #[test_case("keep_metric_names", KeepMetricNames)]
    #[test_case("limit", Limit)]
    #[test_case("offset", Offset)]
    #[test_case("on", On)]
    #[test_case("unless", OpUnless)]
    #[test_case("with", With)]
    #[test_case("without", Without)]
    fn keyword(src: &str, tok: TokenKind) {
        test_tokens!(src, [tok]);
    }

    #[test]
    fn identifier() {
        test_tokens!("foobar123", [Ident=>"foobar123"]);
    }

    #[test]
    fn identifiers() {
        test_tokens!(
          "foo bar123",
          [
            Ident=>"foo",
            Whitespace,
            Ident=>"bar123",
          ]
        );
    }

    #[test]
    fn py_comment() {
        test_tokens!("# hi", [SingleLineHashComment]);
    }

    #[test]
    fn junk() {
        let src = "💩";
        test_tokens!(src, [ErrorInvalidToken]);
    }

    mod golden {
        use super::*;
        use core::fmt;

        #[derive(PartialEq, Eq, Clone, Copy)]
        struct PrettyString<'a>(&'a str);

        impl<'a> PrettyString<'a> {
            #[inline]
            fn new<S: AsRef<str> + ?Sized>(value: &'a S) -> Self {
                PrettyString(value.as_ref())
            }
        }

        impl<'a> fmt::Debug for PrettyString<'a> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(self.0)
            }
        }

        macro_rules! assert_eq {
      ($left:expr, $right:expr $(,$($rest:tt)*)?) => {
        pretty_assertions::assert_eq!(
          PrettyString::new($left),
          PrettyString::new($right)
          $(,$($rest)*)?
        );
      }
    }
    }
}
