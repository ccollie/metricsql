#![forbid(unsafe_code)]
extern crate logos;
extern crate core;
extern crate enquote;
extern crate once_cell;
extern crate phf;
extern crate regex;
extern crate thiserror;

pub mod error;
pub mod ast;
pub mod binaryop;
mod lexer;

pub mod utils {
    use crate::lexer;
    pub use lexer::{ escape_ident, quote, parse_float };
}
pub mod optimizer;
pub mod parser;
