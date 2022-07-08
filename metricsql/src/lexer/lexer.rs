use crate::lexer::TokenKind;
use logos::Logos;
use std::collections::VecDeque;
use std::ops::Range as StdRange;
use text_size::{TextRange, TextSize};

/// A token of metricsql source.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,

    pub range: TextRange,

    _text: String,
}

impl Token {
    #[inline]
    pub fn text(&self) -> String {
        if !self._text.is_empty() {
            return self._text.to_string();
        }
        format!("{}", self.kind).to_string()
    }
}

/// A lexer of metricsql source.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, TokenKind>,
    done: bool,
    peeked: VecDeque<Token>,
}

impl<'a> Lexer<'a> {
    pub fn new(content: &'a str) -> Self {
        Self {
            inner: TokenKind::lexer(content),
            done: false,
            peeked: VecDeque::new(),
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

            Some(kind) => {
                let range = {
                    let StdRange { start, end } = self.inner.span();
                    let start = TextSize::try_from(start).unwrap();
                    let end = TextSize::try_from(end).unwrap();

                    TextRange::new(start, end)
                };

                let _text = match kind {
                    TokenKind::Ident
                    | TokenKind::Duration
                    | TokenKind::Number
                    | TokenKind::QuotedString
                    | TokenKind::SingleLineHashComment => self.inner.slice().to_string(),
                    _ => "".to_string(),
                };
                Some(Token { kind, _text, range })
            }
        }
    }

    pub fn next(&mut self) -> Option<Token> {
        match self.peeked.pop_front() {
            Some(v) => Some(v),
            None => self.read_token(),
        }
    }

    #[inline]
    pub fn peek(&mut self) -> Option<&Token> {
        self.peek_nth(0)
    }

    pub fn peek_nth(&mut self, n: usize) -> Option<&Token> {
        while self.peeked.len() <= n && !self.done {
            if let Some(tok) = self.read_token() {
                self.peeked.push_back(tok);
            }
        }

        self.peeked.get(n)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    #[inline]
    fn next(&mut self) -> Option<Token> {
        match self.peeked.pop_front() {
            Some(v) => Some(v),
            None => self.read_token(),
        }
    }
}
