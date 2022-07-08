use std::{fmt};
use std::error::Error;
use std::fmt::Display;
use thiserror::Error;
use metricsql::parser::ParseError;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, PartialEq, Clone, Error)]
pub enum RuntimeError {
    #[error("Duplicate argument `{0}`")]
    DuplicateArgument(String),
    #[error("Expected number: found `{0}`")]
    InvalidNumber(String),
    #[error("Argument error: {0}")]
    ArgumentError(String),
    #[error(transparent)]
    InvalidArgCount(ArgCountError),
    #[error("{0}")]
    General(String),
    #[error("Invalid regex: {0}")]
    InvalidRegex(String),
    #[error("Aggregate Error: {0}")]
    AggregateError(String),
    #[error("Unknown function `{0}`")]
    UnknownFunction(String),
    #[error(transparent)]
    ParseError(ParseError),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}


impl From<&str> for RuntimeError {
    fn from(message: &str) -> Self {
        RuntimeError::General(String::from(message))
    }
}

impl From<String> for RuntimeError {
    fn from(message: String) -> Self {
        RuntimeError::General(String::from(message))
    }
}

impl<E: Error + 'static> From<(String, E)> for RuntimeError {
    fn from((message, err): (String, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        RuntimeError::General(String::from(msg))
    }
}

impl<E: Error + 'static> From<(&str, E)> for RuntimeError {
    fn from((message, err): (&str, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        RuntimeError::General(String::from(msg))
    }
}

/// Occurs when a function is called with the wrong number of arguments
#[derive(Debug, PartialEq, Clone, Error)]
pub struct ArgCountError {
    pos: Option<usize>,
    min: usize,
    max: usize,
    signature: String
}

impl ArgCountError {
    /// Create a new instance of the error
    ///
    /// # Arguments
    /// * `signature` - Function call signature
    /// * `min` - Smallest allowed number of arguments
    /// * `max` - Largest allowed number of arguments
    pub fn new(signature: &str, min: usize, max: usize) -> Self {
        Self::new_with_index(None, signature, min, max )
    }

    /// Create a new instance of the error at a specific position
    ///
    /// # Arguments
    /// * `pos` - Index at which the error occured
    /// * `signature` - Function call signature
    /// * `min` - Smallest allowed number of arguments
    /// * `max` - Largest allowed number of arguments
    pub fn new_with_index(pos: Option<usize>, signature: &str, min: usize, max: usize) -> Self {
        Self { pos, min, max, signature: signature.to_string() }
    }

    /// Function call signature
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// Smallest allowed number of arguments
    pub fn min(&self) -> usize {
        self.min
    }

    /// Largest allowed number of arguments
    pub fn max(&self) -> usize {
        self.max
    }

    /// Return the location at which the error occured
    pub fn pos(&self) -> Option<usize> {
        self.pos
    }
}

impl Display for ArgCountError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.min == self.max {
            write!(f, "{}: expected {} args", self.signature, self.min)?;
        } else {
            write!(f, "{}: expected {}-{} args", self.signature, self.min, self.max)?;
        }

        if let Some(pos) = self.pos {
            write!(f, " at position {}", pos)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range as StdRange;
    use crate::lexer::TokenKind;

    fn check(
        expected: Vec<TokenKind>,
        found: Option<TokenKind>,
        range: StdRange<u32>,
        output: &str,
    ) {
        let error = RuntimeError {
            expected,
            found,
            range: {
                let start = range.start.into();
                let end = range.end.into();
                TextRange::new(start, end)
            },
            context: "".to_string()
        };

        assert_eq!(format!("{}", error), output);
    }

    #[test]
    fn one_expected_did_find() {
        check(
            vec![TokenKind::Equals],
            Some(TokenKind::Ident),
            10..20,
            "error at 10..20: expected ‘=’, but found identifier",
        );
    }

    #[test]
    fn one_expected_did_not_find() {
        check(
            vec![TokenKind::RParen],
            None,
            5..6,
            "error at 5..6: expected ‘)’",
        );
    }

    #[test]
    fn two_expected_did_find() {
        check(
            vec![TokenKind::Plus, TokenKind::Minus],
            Some(TokenKind::Equals),
            0..1,
            "error at 0..1: expected ‘+’ or ‘-’, but found ‘=’",
        );
    }

    #[test]
    fn multiple_expected_did_find() {
        check(
            vec![
                TokenKind::Number,
                TokenKind::Ident,
                TokenKind::Minus,
                TokenKind::LParen,
            ],
            Some(TokenKind::LetKw),
            100..105,
            "error at 100..105: expected number, identifier, ‘-’ or ‘(’, but found ‘let’",
        );
    }
}