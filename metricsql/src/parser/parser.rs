use crate::lexer::{Lexer, TextSpan, Token, TokenKind, unescape_ident};
use crate::parser::parse_error::{InvalidTokenError, ParseError};
use crate::parser::{ParseErr, ParseResult};


/// parser parses MetricsQL expression.
///
/// preconditions for all parser.parse* funcs:
/// - self.lex.token should point to the first token to parse.
///
/// post-conditions for all parser.parse* funcs:
/// - self.lex.token should point to the next token after the parsed token.
pub struct Parser<'a> {
    pub(super) input: &'a str,
    tokens: Vec<Token<'a>>,
    cursor: usize,
    kind: TokenKind,
    pub(super) parsing_with: bool,
}

impl<'a> Parser<'a> {
    pub(crate) fn from_tokens(tokens: Vec<Token<'a>>) -> Self {
        let tokens: Vec<_> = tokens.into_iter().filter(|x| !x.kind.is_trivia()).collect();

        let kind = if tokens.len() > 0 {
            tokens[0].kind
        } else {
            TokenKind::Eof
        };

        Self {
            input: "",
            cursor: 0,
            tokens,
            parsing_with: false,
            kind
        }
    }

    pub fn new(input: &'a str) -> Self {
        let lexer = Lexer::new(input);
        let tokens = lexer.collect();

        let mut parser = Parser::from_tokens(tokens);
        parser.input = input;
        parser
    }

    pub fn next_token(&mut self) -> Option<&Token<'a>> {
        if self.is_eof() {
            return None
        }

        self.cursor += 1;
        match self.tokens.get(self.cursor) {
            Some(t) => {
                self.kind = t.kind;
                Some(t)
            },
            None => None
        }
    }

    pub fn prev_token(&mut self) -> Option<&Token<'a>> {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        match self.tokens.get(self.cursor) {
            Some(t) => {
                self.kind = t.kind;
                Some(t)
            },
            None => {
                self.kind = TokenKind::ErrorInvalidToken;
                None
            }
        }
    }

    pub fn peek_token(&self) -> Option<&Token<'a>> {
        self.tokens.get(self.cursor)
    }

    pub(crate) fn current_token(&self) -> ParseResult<&Token<'a>> {
        match self.tokens.get(self.cursor) {
            Some(t) => {
                if t.kind.is_error_token() {
                    let error_msg = format!("{}: {}", t.kind, t.text);
                    return Err(ParseError::General(error_msg.to_string()));
                }
                Ok(t)
            },
            None => Err(ParseError::UnexpectedEOF)
        }
    }

    pub fn is_eof(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(crate) fn last_token_range(&self) -> Option<TextSpan> {
        let index = if self.is_eof() {
            self.tokens.len() - 1
        } else {
            self.cursor
        };
        self.tokens.get(index).map(|Token { span, .. }| *span)
    }

    pub(super) fn update_span(&self, span: &mut TextSpan) -> bool {
        if let Some(end_span) = self.last_token_range() {
            span.intersect_with(end_span);
            return true;
        }
        false
    }

    pub(crate) fn expect(&mut self, kind: TokenKind) -> ParseResult<()> {
        if self.at(kind) {
            self.bump();
            Ok(())
        } else {
            Err(self.token_error(&[kind]))
        }
    }

    pub(crate) fn expect_token(&mut self, kind: TokenKind) -> ParseResult<&Token<'a>> {
        self.expect_one_of(&[kind])
    }

    pub(crate) fn expect_one_of(&mut self, kinds: &[TokenKind]) -> ParseResult<&Token<'a>> {
        if self.at_set(kinds) {
            // todo: weirdness to avoid borrowing
            self.cursor += 1;
            let tok = self.tokens.get(self.cursor - 1).unwrap();
            Ok(tok)
        } else {
            Err(self.token_error(kinds))
        }
    }

    pub(crate) fn token_error(&self, expected: &[TokenKind]) -> ParseError {
        let current_token = self.peek_token();

        let (found, range) = if let Some(Token { kind, span, .. }) = current_token {
            (Some(*kind), *span)
        } else {
            // If we’re at the end of the input we use the range of the very last token in the
            // input.
            (None, self.last_token_range().unwrap_or_default())
        };

        let inner = InvalidTokenError::new(expected, found, range);

        ParseError::InvalidToken(inner)
    }

    pub(super) fn parse_ident_list(&mut self) -> ParseResult<Vec<String>> {
        use TokenKind::*;

        self.expect(LeftParen)?;
        self.parse_comma_separated(&[RightParen],|parser| {
            let tok = parser.expect_token(Ident)?;
            Ok(unescape_ident(tok.text))
        })

    }

    /// Parse a comma-separated list of 1+ items accepted by `F`
    pub fn parse_comma_separated<T, F>(&mut self, stop_tokens: &[TokenKind], mut f: F) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Parser<'a>) -> ParseResult<T>,
    {
        let mut values = Vec::with_capacity(4);
        loop {
            if self.at_set(stop_tokens) {
                self.bump();
                break
            }
            let item = f(self)?;
            values.push(item);
            let kind = self.peek_kind();
            if kind == TokenKind::Comma {
                self.bump();
                continue
            } else if stop_tokens.contains(&kind) {
                self.bump();
                break;
            } else {
                let mut expected = Vec::from(stop_tokens);
                expected.insert(0, TokenKind::Comma);
                return Err(self.token_error(&expected));
            }
        }
        Ok(values)
    }

    pub(super) fn bump(&mut self) {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
            if self.cursor < self.tokens.len() {
                self.kind = self.tokens[self.cursor].kind;
            } else {
                self.kind = TokenKind::Eof;
            }
        } else {
            self.kind = TokenKind::Eof;
        }
    }

    pub(super) fn back(&mut self) -> &mut Self {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.kind = self.tokens[self.cursor].kind;
        }
        self
    }

    pub(crate) fn at(&self, kind: TokenKind) -> bool {
        self.peek_kind() == kind
    }

    pub(super) fn at_set(&self, set: &[TokenKind]) -> bool {
        let kind = self.peek_kind();
        set.contains(&kind )
    }

    pub(crate) fn at_end(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    pub(super) fn peek_kind(&self) -> TokenKind {
        if self.at_end() {
            return TokenKind::Eof
        }
        // self.kind
        let tok = self.tokens.get(self.cursor);
        tok.expect("BUG: invalid index out of bounds").kind
    }

}


/// unexpected creates a parser error complaining about an unexpected lexer item.
/// The item that is presented as unexpected is always the last item produced
/// by the lexer.
pub(super) fn unexpected(p: &mut Parser, context: &str, expected: &str, span: Option<TextSpan>) -> ParseError {
    let mut err_msg: String = String::with_capacity(25 + context.len() + expected.len());

    let span = span.unwrap_or_else(|| p.last_token_range().unwrap() );
    err_msg.push_str("unexpected ");
    let text = match p.current_token() {
        Ok(t) => {
            t.text
        },
        Err(_) => {
            "EOF"
        }
    };

    err_msg.push_str(text);

    if !context.is_empty() {
        err_msg.push_str(" in ");
        err_msg.push_str(context)
    }

    if !expected.is_empty() {
        err_msg.push_str(", expected ");
        err_msg.push_str(expected)
    }

    ParseError::Unexpected(
        ParseErr::new(&err_msg, p.input, span)
    )
}

