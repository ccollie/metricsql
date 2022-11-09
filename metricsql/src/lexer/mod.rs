mod duration;
mod lexer;
mod number;
mod tokens;
mod utils;

#[cfg(test)]
mod lexer_tests;

pub(crate) use duration::{duration_value};
pub use number::parse_float;
pub(crate) use lexer::*;
pub use tokens::*;
pub use utils::{escape_ident, is_string_prefix, quote, unescape_ident};

