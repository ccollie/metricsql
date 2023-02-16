mod duration;
mod lexer;
mod number;
mod tokens;
mod utils;

pub(crate) use duration::parse_duration_value;
pub(crate) use lexer::{Lexer, Token};
pub(crate) use tokens::*;

pub use lexer::TextSpan;
pub use number::{get_number_suffix, parse_number};
pub use utils::{escape_ident, extract_string_value, quote, unescape_ident};

#[cfg(test)]
mod lexer_tests;
