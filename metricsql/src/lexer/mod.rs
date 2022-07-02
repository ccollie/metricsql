mod tokens;
mod duration;
mod utils;
mod number;

pub use tokens::*;
pub use duration::{ duration_value, parse_single_duration, scan_duration };
pub(crate) use number::parse_float;
pub(crate) use utils::{
    quote,
    escape_ident,
    unescape_ident,
    is_string_prefix
};