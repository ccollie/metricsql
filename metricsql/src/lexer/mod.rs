mod duration;
mod lexer;
mod number;
mod tokens;
mod utils;

pub(crate) use tokens::*;
pub(crate) use duration::{parse_duration_value};
pub(crate) use lexer::{Lexer, Token};

pub use lexer::TextSpan;
pub use number::{parse_number, get_number_suffix};
pub use utils::{escape_ident, quote, unescape_ident, extract_string_value};

#[cfg(test)]
mod lexer_tests;