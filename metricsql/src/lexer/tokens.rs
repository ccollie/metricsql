use logos::Logos;
use std::fmt::{Display, Formatter};

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
#[logos(subpattern decimal = r"[0-9][_0-9]*")]
#[logos(subpattern hex = r"-?0[xX][0-9a-fA-F][_0-9a-fA-F]*")]
#[logos(subpattern octal = r"0o?[_0-7]*")]
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
