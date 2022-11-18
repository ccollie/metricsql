mod duration;
mod lexer;
mod number;
mod tokens;
mod utils;

pub(crate) use duration::{duration_value};
pub(crate) use lexer::*;

pub use number::parse_float;
pub use tokens::*;
pub use utils::{escape_ident, is_string_prefix, quote, unescape_ident};

#[cfg(test)]
mod lexer_tests;