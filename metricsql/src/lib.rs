#![forbid(unsafe_code)]
extern crate core;
extern crate enquote;
extern crate logos;
extern crate once_cell;
extern crate phf;
extern crate regex;
extern crate serde;
extern crate thiserror;

pub mod error;
pub mod ast;
pub mod binaryop;
mod lexer;

pub mod utils {
    pub use lexer::{escape_ident, parse_float, quote};

    use crate::lexer;
}

pub mod optimizer;
pub mod parser;
pub mod functions;