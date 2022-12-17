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
