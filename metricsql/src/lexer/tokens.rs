#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
extern crate logos;

use alloc::collections::VecDeque;
use core::num::dec2flt::float::RawFloat;
use logos::Logos;
use regex::internal::Input;
use crate::error::Error;
use crate::lexer::parse_float;
use crate::types::BinaryOp as Operator;
use super::duration::duration_value;

static DurationRegex: &str = r"(-?)(?=\d)(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)([DW]))?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?";

// Note: callbacks can return `Option` or `Result`
fn number(lex: &mut Lexer) -> Option<f64> {
    let slice = lex.slice();
    return parse_float(slice);
}

fn lex_operator(lex: &mut Lexer) -> Result<Operator, Error> {
    let slice = lex.slice();
    Operator::try_from(slice)
}

#[derive(Logos, Debug, PartialEq)]
enum RawToken<'a> {
    #[token("and", ignore(ascii_case))]
    And,

    #[token("atan2", ignore(ascii_case))]
    OpAtan2,

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

    #[regex(DurationRegex)]
    Duration,

    #[regex(r"[_a-zA-Z][_a-zA-Z0-9]*", |lex| lex.slice())]
    Ident(&'a str),

    #[regex("[-|+]?[i|I][n|N][f|F]", number)]
    #[token("[-|+]?[n|N][a|A][n|N]", number)]
    #[regex("0x[0-9a-fA-F]+")]
    #[regex(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")]
    Number(f64),

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

    #[token("=~")]
    RegexEqual,

    #[token("!~")]
    RegexNotEqual,

    #[regex(r"[!\$:~\+\-&\|\^=<>\*/%]+", lex_operator)]
    Op(Operator),

    #[regex("'(?s:[^'\\\\]|\\\\.)*'", |lex| lex.slice())]
    #[regex("\"(?s:[^\"\\\\]|\\\\.)*\"", |lex| lex.slice())]
    QuotedString,

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

    #[error]
    Error,
}

macro_rules! token_enum {
  (
    $(#[$($enum_m:tt)*])*
    pub enum $name:ident {
      $($case:ident $(#[$($m:tt)*])*)*
    }
  ) => {
    $(#[$($enum_m)*])*
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub enum $name {
      $(
        $(#[$($m)*])*
        $case,
      )*
    }
  };
}

token_enum! {
  /// Token kind.
  #[derive(Display, Debug)]
  pub enum TokenKind {
    // keywords
    And                                /// `and`
    By                                 /// `by`
    Bool                               /// `bool`
    Default                            /// `default`
    GroupLeft                          /// `group_left`
    GroupRight                         /// `group_right`
    If                                 /// `if`
    IfNot                              /// `ifNot`
    Ignoring                           /// `ignoring`
    KeepMetricNames                    /// `keep_metric_names`
    Limit                              /// `limit`
    On                                 /// `on`
    Offset                             /// `offset`
    Or                                 /// `or`
    Unless                             /// `unless`
    With                               /// `with`
    Without                            /// `without`

    // variable tokens
    Duration                                  /// duration token
    Ident                                     /// identifier token
    Number                                    /// number token

    // symbols
    At                                  /// '@'
    LeftBrace                                 /// `{`
    RightBrace                                /// `}`
    LeftBracket                               /// `[`
    RightBracket                              /// `]`
    Colon                                     /// `:`
    Comma                                    /// `,`
    Dot                                         /// `.`
    LeftParen                                   /// `(`
    RightParen                                  /// `)`
    SemiColon                                   /// `;`
    Equal                                       /// '='

    // operators
    OpAtan2                                   /// `atan2`
    OpMul                                     /// `*`
    OpDiv                                     /// `/`
    OpIf                                      /// 'if'
    OpIfNot                                   /// 'ifNot'
    OpMod                                     /// `%`
    OpPlus                                    /// `+`
    OpMinus                                   /// `-`
    OpLessThan                                /// `<`
    OpGreaterThan                             /// `>`
    OpLessThanOrEqual                         /// `<=`
    OpGreaterThanOrEqual                      /// `>=`
    OpEqual                                   /// `==`
    OpNotEqual                                /// `!=`
    OpPow                                     /// `^`
    OpRegexEqual                              /// `=~`
    OpRegexNotEqual                           /// `!~`
    OpAnd                                     /// `and`
    OpOr                                      /// `or`
    OpUnless                                  /// 'unless'

    // strings
    QuotedString                              ///  single or double quotes string

    Whitespace                                /// any whitespace
    SingleLineHashComment                     /// `# comment`
    BlockComment                              /// `/* comment */`

    // string errors
    ErrorStringDoubleQuotedUnterminated       /// unterminated double quoted string
    ErrorStringSingleQuotedUnterminated       /// unterminated single quoted string
    ErrorStringMissingQuotes

    // number errors
    ErrorNumJunkAfterDecimalPoint             /// unexpected character after decimal point
    ErrorNumJunkAfterExponent                 /// unexpected character after exponent
    ErrorNumJunkAfterExponentSign             /// unexpected character after exponent sign

    // comment errors
    ErrorCommentUnterminated                  /// unterminated comment

    // other
    ErrorUnknownOperator                      /// unknown operator
    ErrorInvalidToken                         /// invalid token
  }
}

impl TokenKind {
    pub fn is_trivia(self) -> bool {
        matches!(self, Self::Whitespace | Self::SingleLineHashComment)
    }

    pub fn is_error(self) -> bool {
        use TokenKind::*;

        match self {
            ErrorStringDoubleQuotedUnterminated
            | ErrorStringSingleQuotedUnterminated
            | ErrorNumJunkAfterDecimalPoint
            | ErrorNumJunkAfterExponent
            | ErrorNumJunkAfterExponentSign
            | ErrorUnknownOperator
            | ErrorStringMissingQuotes
            | ErrorInvalidToken => true,
            _ => false,
        }
    }
    
    pub fn is_operator(self) -> bool {
        use TokenKind::*;
        
        match self {
            OpAtan2 
            | OpMul
            | OpDiv
            | OpIf
            | OpIfNot
            | OpMod
            | OpPlus
            | OpMinus
            | OpLessThan
            | OpGreaterThan
            | OpLessThanOrEqual
            | OpGreaterThanOrEqual
            | OpEqual
            | OpNotEqual
            | OpPow
            | OpAnd 
            | OpOr
            | OpUnless => true,
            _ => false
        }
    }

    #[inline]
    pub fn is_comparison_op(self) -> bool {
        use TokenKind::*;
        
        match self {
            OpEqual 
            | OpNotEqual 
            | OpGreaterThanOrEqual 
            | OpGreaterThan 
            | OpLessThanOrEqual
            | OpLessThan => true,
            _ => false
        }
    }

    #[inline]
    pub fn is_rollup_start(self) -> bool {
        use TokenKind::*;

        match self {
            Offset | At | LeftBracket => true,
            _ => false
        }
    }

    #[inline]
    pub fn is_group_modifier(self) -> bool {
        use TokenKind::*;
        match self {
            On | Ignoring => true,
            _ => false
        }
    }

    #[inline]
    pub fn is_join_modifier(self) -> bool {
        use TokenKind::*;
        match self {
            GroupLeft | GroupRight => true,
            _ => false
        }
    }

    pub fn is_aggregate_modifier(self) -> bool {
        use TokenKind::*;
        match self {
            By | Without => true,
            _ => false
        }
    }
}

/// A token of metricsql source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token<'a> {
    /// The kind of token.
    pub kind: TokenKind,

    pub text: &'a str,

    pub token: RawToken<'a>,

    /// The token value.
    pub len: u32,
}

/// A lexer of metricsql source.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, RawToken<'a>>,
    done: bool,
    peeked: VecDeque<Token<'a>>,
    current: Option<Token<'a>>,
    #[cfg(debug_assertions)]
    len: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(content: &'a str) -> Self {
        Self {
            inner: RawToken::lexer(content),
            done: false,
            peeked: VecDeque::new(),
            current: None,
            #[cfg(debug_assertions)]
            len: 0,
        }
    }

    fn read_token(&mut self) -> Option<Token> {
        if self.done {
            return None;
        }

        match self.inner.next() {
            None => {
                self.done = true;
                None
            }

            Some(raw) => {
                let len = self.inner.slice().len() as u32;

                #[cfg(debug_assertions)]
                {
                    self.len += len;
                    assert_eq!(self.len, self.inner.span().end as u32);
                }

                self.current = Some(self.to_token(raw, len));
                *self.current
            }
        }
    }

    pub fn next(&mut self) -> Option<Token> {
        match self.peeked.pop_front() {
            Some(v) => {
                self.current = Some(v);
                self.current
            },
            None => self.read_token(),
        }
    }

    pub fn span(&self) -> Logos::Span {
        self.inner.span()
    }

    pub fn token(&self) -> Option<Token<'a>> {
       self.current
    }

    fn to_token(&self, raw: RawToken, len: u32) -> Token {
        let kind = match raw {
            RawToken::And => TokenKind::OpAnd,
            RawToken::OpAtan2 => TokenKind::OpAtan2,
            RawToken::By => TokenKind::By,
            RawToken::Bool => TokenKind::Bool,
            RawToken::Equal => TokenKind::Equal,
            RawToken::Default => TokenKind::Default,
            RawToken::GroupLeft => TokenKind::GroupLeft,
            RawToken::GroupRight => TokenKind::GroupRight,
            RawToken::If => TokenKind::If,
            RawToken::IfNot => TokenKind::IfNot,
            RawToken::Ignoring => TokenKind::Ignoring,
            RawToken::KeepMetricNames => TokenKind::KeepMetricNames,
            RawToken::Limit => TokenKind::Limit,
            RawToken::Offset => TokenKind::Offset,
            RawToken::On => TokenKind::On,
            RawToken::Or => TokenKind::Or,
            RawToken::Unless => TokenKind::Unless,
            RawToken::With => TokenKind::With,
            RawToken::Without => TokenKind::Without,
            RawToken::Duration => TokenKind::Duration,
            RawToken::Ident(..) => TokenKind::Ident,
            RawToken::Number(n) => TokenKind::Number,
            RawToken::RegexEqual => TokenKind::OpRegexEqual,
            RawToken::RegexNotEqual => TokenKind::OpRegexNotEqual,
            RawToken::At => TokenKind::At,
            RawToken::LeftBrace => TokenKind::LeftBrace,
            RawToken::RightBrace => TokenKind::RightBrace,
            RawToken::LeftBracket => TokenKind::LeftBracket,
            RawToken::RightBracket => TokenKind::RightBracket,
            RawToken::Colon => TokenKind::Colon,
            RawToken::Comma => TokenKind::Comma,
            RawToken::Dot => TokenKind::Dot,
            RawToken::LeftParen => TokenKind::LeftParen,
            RawToken::RightParen => TokenKind::RightParen,
            RawToken::SemiColon => TokenKind::SemiColon,
            RawToken::Op(Operator::Mul) => TokenKind::OpMul,
            RawToken::Op(Operator::Div) => TokenKind::OpDiv,
            RawToken::Op(Operator::Mod) => TokenKind::OpMod,
            RawToken::Op(Operator::Add) => TokenKind::OpPlus,
            RawToken::Op(Operator::Sub) => TokenKind::OpMinus,
            RawToken::Op(Operator::Lt) => TokenKind::OpLessThan,
            RawToken::Op(Operator::Gt) => TokenKind::OpGreaterThan,
            RawToken::Op(Operator::Lte) => TokenKind::OpLessThanOrEqual,
            RawToken::Op(Operator::Gte) => TokenKind::OpGreaterThanOrEqual,
            RawToken::Op(Operator::Eql) => TokenKind::OpEqual,
            RawToken::Op(Operator::Neq) => TokenKind::OpNotEqual,
            RawToken::Op(Operator::Pow) => TokenKind::OpPow,
            RawToken::Op(Operator::And) => TokenKind::OpAnd,
            RawToken::Op(Operator::Or) => TokenKind::OpOr,
            RawToken::QuotedString => TokenKind::QuotedString,
            RawToken::Whitespace => TokenKind::Whitespace,
            RawToken::SingleLineHashComment => TokenKind::SingleLineHashComment,
            RawToken::ErrorNumJunkAfterDecimalPoint => TokenKind::ErrorNumJunkAfterDecimalPoint,
            RawToken::ErrorNumJunkAfterExponent => TokenKind::ErrorNumJunkAfterExponent,
            RawToken::ErrorNumJunkAfterExponentSign => TokenKind::ErrorNumJunkAfterExponentSign,
            RawToken::ErrorStringDoubleQuotedUnterminated => {
                TokenKind::ErrorStringDoubleQuotedUnterminated
            }
            RawToken::ErrorStringSingleQuotedUnterminated => {
                TokenKind::ErrorStringSingleQuotedUnterminated
            }
            RawToken::ErrorStringMissingQuotes => TokenKind::ErrorStringMissingQuotes,
            // RawToken::Op(Operator::Unknown) => TokenKind::ErrorUnknownOperator,
            RawToken::Error => TokenKind::ErrorInvalidToken,
        };

        Token { kind, len, token: &raw, text: self.slice() }
    }

    #[inline]
    pub fn peek(&mut self) -> Option<Token> {
        self.peek_nth(0)
    }

    pub fn peek_nth(&mut self, n: usize) -> Option<Token> {
        while self.peeked.len() <= n && !self.done {
            if let Some(tok) = self.read_token() {
                self.peeked.push_back(tok);
            }
        }

        self.peeked.get(n).copied()
    }

    pub fn is_eof(&self) -> bool {
        self.done
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.peeked.pop_front() {
            Some(v) => Some(v),
            None => self.read_token(),
        }
    }
}

/// Tokenize a metricsql string into a list of tokens.
pub fn tokenize<'a>(content: &'a str) -> impl Iterator<Item=Token> + 'a {
    Lexer::new(content)
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
    fn c_comment() {
        test_tokens!("// hi", [SingelLineSlashComment]);
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