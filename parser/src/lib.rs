#![forbid(unsafe_code)]
extern crate core;
extern crate enquote;
extern crate logos;
extern crate num_traits;
extern crate phf;
extern crate regex;
extern crate serde;
extern crate strum;
extern crate strum_macros;
extern crate thiserror;
extern crate xxhash_rust;

pub mod ast;
pub mod binaryop;
pub mod common;
pub mod functions;
pub mod label;
pub mod optimizer;
pub mod parser;

pub mod prelude {
    pub use crate::ast::*;
    pub use crate::binaryop::*;
    pub use crate::common::*;
    pub use crate::functions::*;
    pub use crate::label::*;
    pub use crate::optimizer::*;
    pub use crate::parser::*;
}
