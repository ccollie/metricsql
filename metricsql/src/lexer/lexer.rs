use std::collections::VecDeque;
use crate::lexer::TokenKind;
use text_size::{TextRange, TextSize};
use std::ops::Range as StdRange;
use logos::Logos;

/// A token of metricsql source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token<'a> {
    /// The kind of token.
    pub kind: TokenKind,

    pub text: &'a str,

    pub range: TextRange,
}

/// A lexer of metricsql source.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, TokenKind>,
    done: bool,
    peeked: VecDeque<Token<'a>>,
    current: Option<Token<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(content: &'a str) -> Self {
        Self {
            inner: TokenKind::lexer(content),
            done: false,
            peeked: VecDeque::new(),
            current: None,
        }
    }

    fn read_token(&mut self) -> Option<Token<'a>> {
        if self.done {
            return None;
        }

        match self.inner.next() {
            None => {
                self.done = true;
                None
            }

            Some(..) => {
                let kind = self.inner.next()?;
                let text = self.inner.slice();

                let range = {
                    let StdRange { start, end } = self.inner.span();
                    let start = TextSize::try_from(start).unwrap();
                    let end = TextSize::try_from(end).unwrap();

                    TextRange::new(start, end)
                };

                self.current = Some(Token { kind, text, range });
                self.current
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

    pub fn token(&self) -> Option<Token<'a>> {
        self.current
    }

    #[inline]
    pub fn peek(&mut self) -> Option<Token> {
        self.peek_nth(0)
    }

    pub fn peek_nth(&mut self, n: usize) -> Option<Token<'a>> {
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
    fn next(&mut self) -> Option<Token<'a>> {
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
