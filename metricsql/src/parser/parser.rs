use super::expand::{expand_with_expr};
use crate::ast::{DurationExpr, Expr, ParensExpr, WithArgExpr};
use crate::common::StringExpr;
use crate::parser::expand::resolve_ident;
use crate::parser::expr::{handle_escape_ident, parse_expression};
use crate::parser::symbol_provider::{HashMapSymbolProvider, SymbolProviderRef};
use crate::parser::tokens::Token;
use crate::parser::{
    extract_string_value, invalid_token_error, parse_duration_value, parse_number, syntax_error,
    ParseError, ParseResult,
};
use crate::prelude::unescape_ident;
use logos::{Logos, Span};
use std::sync::Arc;

/// A token of MetricSql source.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithLocation<'source> {
    /// The kind of token.
    pub kind: Token,
    pub text: &'source str,
    pub span: Span,
}

impl<'source> TokenWithLocation<'source> {
    pub fn len(&self) -> usize {
        self.text.len()
    }
}

/// parser parses MetricsQL expression.
///
/// preconditions for all parser.parse* funcs:
/// - self.tokens[self.cursor] should point to the first token to parse.
///
/// post-conditions for all parser.parse* funcs:
/// - self.token[self.cursor] should point to the next token after the parsed token.
pub struct Parser<'a> {
    tokens: Vec<TokenWithLocation<'a>>,
    symbol_provider: SymbolProviderRef,
    pub(super) cursor: usize,
    pub(super) needs_expansion: bool,
    pub(super) with_stack: Vec<Vec<WithArgExpr>>,
}

impl<'a> Parser<'a> {
    pub fn tokenize(input: &'a str) -> ParseResult<Vec<TokenWithLocation<'a>>> {
        let mut lexer: logos::Lexer<'a, Token> = Token::lexer(input);

        let mut tokens = Vec::with_capacity(16); // todo: pre-size
        loop {
            match lexer.next() {
                Some(tok) => match tok {
                    Ok(tok) => {
                        tokens.push(TokenWithLocation {
                            kind: tok,
                            text: lexer.slice(),
                            span: lexer.span(),
                        });
                    }
                    Err(e) => return Err(e),
                },
                _ => break,
            }
        }

        Ok(tokens)
    }

    pub fn new(input: &'a str) -> ParseResult<Self> {
        let tokens = Self::tokenize(input)?;

        Ok(Self {
            cursor: 0,
            tokens,
            needs_expansion: false,
            with_stack: vec![],
            symbol_provider: Arc::new(HashMapSymbolProvider::new()),
        })
    }

    pub fn from_tokens(tokens: Vec<TokenWithLocation<'a>>) -> Self {
        Self {
            cursor: 0,
            tokens,
            needs_expansion: false,
            with_stack: vec![],
            symbol_provider: Arc::new(HashMapSymbolProvider::new()),
        }
    }

    // next
    pub fn next(&mut self) -> Option<&TokenWithLocation<'a>> {
        if self.is_eof() {
            return None;
        }
        self.cursor += 1;
        self.tokens.get(self.cursor)
    }

    pub(super) fn next_token(&mut self) -> Option<&TokenWithLocation<'a>> {
        if self.is_eof() {
            return None;
        }
        self.cursor += 1;
        self.tokens.get(self.cursor)
    }

    pub fn prev_token(&mut self) -> Option<&TokenWithLocation<'a>> {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        self.tokens.get(self.cursor)
    }

    pub fn peek_token(&self) -> Option<&TokenWithLocation<'a>> {
        self.tokens.get(self.cursor)
    }

    pub(crate) fn current_token(&self) -> ParseResult<&TokenWithLocation<'a>> {
        match self.tokens.get(self.cursor) {
            Some(t) => Ok(t),
            None => Err(ParseError::UnexpectedEOF),
        }
    }

    pub fn is_eof(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(super) fn last_token_range(&self) -> Option<Span> {
        let index = if self.is_eof() {
            if self.tokens.len() > 0 {
                self.tokens.len() - 1
            } else {
                0
            }
        } else {
            self.cursor
        };
        self.tokens
            .get(index)
            .map(|TokenWithLocation { span, .. }| span.clone())
    }

    pub(super) fn expect(&mut self, kind: &Token) -> ParseResult<()> {
        if self.at(kind) {
            self.bump();
            Ok(())
        } else {
            Err(self.token_error(&[kind.clone()]))
        }
    }

    pub(crate) fn expect_one_of(&mut self, kinds: &[Token]) -> ParseResult<&TokenWithLocation<'a>> {
        if self.at_set(kinds) {
            let prev = self.cursor;
            self.cursor += 1;
            let tok = self.tokens.get(prev).unwrap();
            Ok(tok)
        } else {
            Err(self.token_error(kinds))
        }
    }

    pub(super) fn expect_identifier(&mut self) -> ParseResult<String> {
        let tok = self.expect_one_of(&[Token::Identifier])?;
        let ident = if tok.text.contains(r#"\"#) {
            unescape_ident(tok.text)
        } else {
            tok.text.to_string()
        };
        Ok(ident)
    }

    pub(super) fn token_error(&self, expected: &[Token]) -> ParseError {
        let current_token = self.peek_token();

        if let Some(TokenWithLocation { kind, span, .. }) = current_token {
            invalid_token_error(expected, Some(kind.clone()), span, "".to_string())
        } else {
            // If we’re at the end of the input we use the range of the very last token in the input.
            let span = self.last_token_range().unwrap_or_default();
            invalid_token_error(expected, None, &span, "".to_string())
        }
    }

    pub(super) fn syntax_error(&self, msg: &str) -> ParseError {
        let span = self.last_token_range().unwrap_or_default();
        syntax_error(msg, &span, "".to_string())
    }

    pub(super) fn parse_number(&mut self) -> ParseResult<f64> {
        let token = self.current_token()?;
        let value = parse_number(&token.text).map_err(|_| {
            let msg = format!("expect number, got {}", token.text);
            syntax_error(&msg, &token.span, "".to_string())
        })?;

        self.bump();

        Ok(value)
    }

    pub fn parse_duration(&mut self) -> ParseResult<DurationExpr> {
        use Token::*;

        let mut requires_step = false;
        let token = self.expect_one_of(&[Number, Duration])?;
        let last_ch: char = token.text.chars().last().unwrap();

        let value = match token.kind {
            Number => {
                // there is a bit of ambiguity between a Number with a unit and a Duration in the
                // case of tokens with the suffix 'm'. For example, does 60m mean 60 minutes or
                // 60 million. We accept Duration here to deal with that special case
                if last_ch == 'm' || last_ch == 'M' {
                    // treat as minute
                    parse_duration_value(token.text, 1)?
                } else {
                    parse_number(token.text)? as i64
                }
            }
            Duration => {
                requires_step = last_ch == 'i' || last_ch == 'I';
                parse_duration_value(token.text, 1)?
            }
            _ => unreachable!("parse_duration"),
        };

        Ok(DurationExpr {
            value,
            requires_step,
        })
    }

    /// returns positive duration in milliseconds for the given s
    /// and the given step.
    ///
    /// Duration in s may be combined, i.e. 2h5m or 2h-5m.
    ///
    /// Error is returned if the duration in s is negative.
    pub fn parse_positive_duration(&mut self) -> ParseResult<DurationExpr> {
        // Verify the duration in seconds without explicit suffix.
        let duration = self.parse_duration()?;
        let val = duration.value(1);
        if val < 0 {
            Err(ParseError::InvalidDuration(duration.to_string()))
        } else {
            Ok(duration)
        }
    }

    pub fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use Token::*;

        self.expect(&LeftParen)?;
        self.parse_comma_separated(&[RightParen], |parser| Ok(parser.expect_identifier()?))
    }

    pub(super) fn parse_arg_list(&mut self) -> ParseResult<Vec<Expr>> {
        use Token::*;
        self.expect(&LeftParen)?;
        self.parse_comma_separated(&[RightParen], parse_expression)
    }

    /// Parse a comma-separated list of 1+ items accepted by `f`
    pub fn parse_comma_separated<T, F>(
        &mut self,
        stop_tokens: &[Token],
        mut f: F,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Parser<'a>) -> ParseResult<T>,
    {
        let mut values = Vec::with_capacity(4);
        loop {
            if self.at_set(stop_tokens) {
                self.bump();
                break;
            }
            let item = f(self)?;
            values.push(item);

            let kind = self.peek_kind();
            if kind == Token::Comma {
                self.bump();
                continue;
            } else if stop_tokens.contains(&kind) {
                self.bump();
                break;
            } else {
                let mut expected = Vec::from(stop_tokens);
                expected.insert(0, Token::Comma);
                return Err(self.token_error(&expected));
            }
        }
        Ok(values)
    }

    pub(super) fn parse_string_expression(&mut self) -> ParseResult<StringExpr> {
        use Token::*;

        let mut tok = self.current_token()?;
        let mut result = StringExpr::default();
        // if we have no symbols or not in a WITH, we don't accept identifiers since we can't resolve
        // them
        let accept_identifiers = self.can_lookup();

        loop {
            match &tok.kind {
                StringLiteral => {
                    let value = extract_string_value(tok.text)?;
                    if !value.is_empty() {
                        result.push_str(&value);
                    }
                }
                Identifier => {
                    if !accept_identifiers {
                        let msg = "identifiers are not allowed in this context";
                        return Err(self.syntax_error(msg));
                    }
                    let ident = handle_escape_ident(tok.text);
                    result.push_ident(&ident);
                    self.needs_expansion = true;
                }
                _ => {
                    return Err(self.token_error(&[StringLiteral, Identifier]));
                }
            }

            if let Some(token) = self.next_token() {
                tok = token;
            } else {
                break;
            }

            if tok.kind != OpPlus {
                break;
            }

            // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
            if let Some(token) = self.next_token() {
                tok = token;
            } else {
                break;
            }

            if tok.kind == StringLiteral {
                // "s1" + "s2"
                continue;
            }

            if tok.kind != Identifier {
                // "s" + unknownToken
                self.back();
                break;
            }

            // Look after ident
            match self.next_token() {
                None => break,
                Some(t) => {
                    tok = t;
                    if tok.kind == LeftParen || tok.kind == LeftBrace {
                        self.back();
                        self.back();
                        // `"s" + m(` or `"s" + m{`
                        break;
                    }
                }
            }

            // "s" + ident
            tok = self.prev_token().unwrap();
        }

        Ok(result)
    }

    pub(super) fn parse_parens_expr(&mut self) -> ParseResult<Expr> {
        let list = self.parse_arg_list()?;
        Ok(Expr::Parens(ParensExpr::new(list)))
    }

    pub(super) fn bump(&mut self) {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
        }
    }

    pub(super) fn back(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub(super) fn at(&self, kind: &Token) -> bool {
        let actual = self.peek_kind();
        &actual == kind
    }

    pub(super) fn at_set(&self, set: &[Token]) -> bool {
        let kind = self.peek_kind();
        set.contains(&kind)
    }

    pub(crate) fn at_end(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(super) fn peek_kind(&self) -> Token {
        if self.at_end() {
            return Token::Eof;
        }
        // self.kind
        let tok = self.tokens.get(self.cursor);
        tok.expect("BUG: invalid index out of bounds").kind
    }

    pub(super) fn is_parsing_with(&self) -> bool {
        !self.with_stack.is_empty()
    }

    pub(super) fn has_symbols(&self) -> bool {
        self.symbol_provider.size() > 0
    }

    pub(super) fn can_lookup(&self) -> bool {
        self.is_parsing_with() || self.has_symbols()
    }

    pub(super) fn lookup_with_expr(&self, name: &str) -> Option<&WithArgExpr> {
        for frame in self.with_stack.iter().rev() {
            if let Some(expr) = frame.iter().find(|x| x.name == name) {
                return Some(expr);
            }
        }
        None
    }

    pub(super) fn resolve_ident(&self, ident: &str, args: &[Expr]) -> ParseResult<Option<Expr>> {
        let empty = vec![];
        let was = self.with_stack.last().unwrap_or(&empty);
        let expr = resolve_ident(&self.symbol_provider, was, ident, args)?;
        Ok(expr)
    }

    pub(super) fn expand_if_needed(&self, expr: Expr) -> ParseResult<Expr> {
        if self.needs_expansion {
            let was: Vec<WithArgExpr> = vec![];
            expand_with_expr(&self.symbol_provider, &was, &expr)
        } else {
            Ok(expr)
        }
    }
}
