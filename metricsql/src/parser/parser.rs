use logos::{Logos, Span};
use crate::ast::{Expr, WithArgExpr};
use crate::parser::expand_expr::expand_with_expr;
use crate::parser::{invalid_token_error, ParseError, ParseResult, syntax_error};
use crate::parser::tokens::Token;
use crate::prelude::unescape_ident;


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
    pub(super) cursor: usize,
    pub(super) template_parsing_depth: usize,
    pub(super) needs_expansion: bool,
    pub(super) with_stack: Vec<Vec<WithArgExpr>>,
}

impl<'a> Parser<'a> {
    pub fn tokenize(input: &'a str) -> ParseResult<Vec<TokenWithLocation<'a>>> {
        let mut lexer: logos::Lexer<'a, Token> = Token::lexer(input);

        let mut tokens = Vec::with_capacity(16); // todo: pre-size
        loop {
            match lexer.next() {
                Some(tok) => {
                    match tok {
                        Ok(tok) => {
                            tokens.push(TokenWithLocation {
                                kind: tok,
                                text: lexer.slice(),
                                span: lexer.span(),
                            });
                        }
                        Err(e) => return Err(e)
                    }
                }
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
            template_parsing_depth: 0,
            needs_expansion: false,
            with_stack: vec![],
        })
    }

    pub fn from_tokens(tokens: Vec<TokenWithLocation<'a>>) -> Self {
        Self {
            cursor: 0,
            tokens,
            template_parsing_depth: 0,
            needs_expansion: false,
            with_stack: vec![],
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

    pub fn next_token(&mut self) -> Option<&TokenWithLocation<'a>> {
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
        self.tokens.get(index).map(|TokenWithLocation { span, .. }| span.clone())
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
            // weirdness to avoid borrowing
            self.cursor += 1;
            let tok = self.tokens.get(self.cursor - 1).unwrap();
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

    pub(super) fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use Token::*;

        self.expect(&LeftParen)?;
        self.parse_comma_separated(&[RightParen], |parser| {
            Ok(parser.expect_identifier()?)
        })
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

    pub(super) fn lookup_with_expr(&self, name: &str) -> Option<&WithArgExpr> {
        for frame in self.with_stack.iter().rev() {
            if let Some(expr) = frame.iter().find(|x| x.name == name) {
                return Some(expr);
            }
        }
        None
    }

    pub(super) fn resolve_value(&self, name: &str) -> Option<&WithArgExpr> {
        if let Some(arg) = self.lookup_with_expr(name) {
            if !arg.is_function {
                return Some(arg);
            }
        }
        None
    }

    pub(super) fn resolve_template_function(&self, name: &str) -> Option<&WithArgExpr> {
        if let Some(arg) = self.lookup_with_expr(name) {
            if arg.is_function {
                return Some(arg);
            }
        }
        None
    }

    pub(super) fn expand_if_needed(&self, expr: Expr) -> ParseResult<Expr> {
        if self.needs_expansion {
            let resolve_fn = |name: &str| {
                if let Some(found) = self.resolve_value(name) {
                    return Some(found.expr.clone()); // todo: use lifetimes !!!!!!
                }
                None
            };
            expand_with_expr(expr, resolve_fn)
        } else {
            Ok(expr)
        }
    }
}