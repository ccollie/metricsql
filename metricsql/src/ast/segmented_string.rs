use std::fmt;
use std::fmt::{Display, Formatter};
use crate::lexer::TextSpan;
use serde::{Serialize, Deserialize};
use crate::prelude::ParseResult;

#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub enum StringSegment {
    Literal(String),
    Ident(String)
}

impl Default for StringSegment {
    fn default() -> Self {
        StringSegment::Literal(String::new())
    }
}

impl Display for StringSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            StringSegment::Literal(lit) => write!(f, "{}", enquote::enquote('"', lit))?,
            StringSegment::Ident(ident) => write!(f, "{}", ident)?
        }
        Ok(())
    }
}

/// SegmentedString represents string segmented expression (used in WITH expressions).
/// Format is (segments, isLiteralOnly)
#[derive(Debug, Clone, Hash, PartialEq, Serialize, Deserialize)]
pub struct SegmentedString(Vec<StringSegment>, bool);

impl SegmentedString {
    pub fn new<S: Into<String>, TS: Into<TextSpan>>(s: S) -> Self {
        let segments = vec![ StringSegment::Literal(s.into()) ];
        SegmentedString(segments, true)
    }

    pub fn with_segment_capacity(initial_capacity: usize) -> Self {
        SegmentedString(Vec::with_capacity(initial_capacity), true)
    }

    pub fn set_from_string<S: Into<String>>(&mut self, s: S) {
        if self.0.len() == 1 {
            if let Some(StringSegment::Literal(elem)) = self.0.first_mut() {
                elem.clear();
                elem.push_str(s.into().as_str())
            }
        }
        self.0.clear();
        self.0.push( StringSegment::Literal(s.into()));
        self.1 = true
    }

    pub fn push_str(&mut self, tok: &str) {
        if let Some(last) = self.0.last_mut() {
            match last {
                StringSegment::Literal(value) => {
                    value.push_str(tok);
                    self.1 = self.0.len() == 1;
                }
                _ => {
                    self.0.push(StringSegment::Literal(tok.to_string()));
                }
            }
        } else {
            self.0.push(StringSegment::Literal(tok.to_string()));
        }
    }

    pub fn push_ident(&mut self, tok: &str) {
        self.0.push(StringSegment::Ident(tok.to_string()));
        self.1 = false;
    }

    pub fn clear(&mut self) {
        self.0.clear();
        self.1 = true;
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn segment_count(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_expanded(&self) -> bool {
        !self.1 // is literal only
    }

    pub(crate) fn is_literal_only(&self) -> bool {
        self.1
    }

    pub fn resolve<F>(&self, f: F) -> ParseResult<String>
        where F: FnMut(&str) -> ParseResult<&str> {
        if self.1 {
            if let Some(first) = self.0.first() {
                match first {
                    StringSegment::Literal(lit) => return Ok(lit.clone()),
                    _ => panic!("BUG: string segment should be all literal")
                }
            } else {
                return Ok("".to_string())
            }
        }
        let min_capacity = self.0.iter().fold(0, |acc, s| {
            let res = match s {
                StringSegment::Literal(lit) => lit.len(),
                StringSegment::Ident(_) => 4, // todo: proper named constant
            };
            acc + res
        });

        let mut res = String::with_capacity(min_capacity);
        for s in self.0.iter() {
            match s {
                StringSegment::Literal(lit) => res.push_str(lit),
                StringSegment::Ident(id) => {
                    let substitute = f(id)?;
                    res.push_str(substitute);
                },
            };
        }

        Ok(res)
    }
}

impl Default for SegmentedString {
    fn default() -> Self {
        SegmentedString(vec![], true)
    }
}

impl From<String> for SegmentedString {
    fn from(s: String) -> Self {
        let segment = StringSegment::Literal(s);
        SegmentedString {
            0: vec![segment],
            1: true,
        }
    }
}

impl From<&str> for SegmentedString {
    fn from(s: &str) -> Self {
        Self::from(s.to_string())
    }
}

impl Display for SegmentedString {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for (i, segment) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", segment)?;
        }
        Ok(())
    }
}
