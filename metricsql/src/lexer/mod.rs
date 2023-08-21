pub use duration::parse_duration_value;
pub use lexer::TextSpan;
pub(crate) use lexer::{Lexer, Token};
pub use number::{get_number_suffix, parse_number};
pub(crate) use tokens::*;
pub use utils::{escape_ident, extract_string_value, quote, unescape_ident};

mod duration;
mod lexer;
mod number;
mod tokens;
mod utils;

#[cfg(test)]
mod lexer_tests;
