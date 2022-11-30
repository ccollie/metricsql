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
    use crate::lexer;
    pub use lexer::{escape_ident, parse_float, quote};
}

pub mod parser;
pub mod functions;
pub mod transform;

pub use lexer::TextSpan;

pub mod prelude {
    use crate::ast;
    use crate::lexer;
    use crate::binaryop;
    use crate::functions;
    use crate::parser;
    use crate::transform;

    pub use ast::*;
    pub use binaryop::*;
    pub use functions::*;
    pub use lexer::TextSpan;
    pub use parser::*;
    pub use transform::*;
}