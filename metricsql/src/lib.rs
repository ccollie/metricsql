#![forbid(unsafe_code)]
extern crate core;
extern crate enquote;
extern crate logos;
extern crate phf;
extern crate regex;
extern crate serde;
extern crate thiserror;
#[macro_use]
extern crate tinyvec;
#[cfg(feature = "xxh64")]
extern crate xxhash_rust;

pub mod ast;
pub mod binaryop;
pub mod common;
pub mod functions;
pub mod parser;

pub mod optimize {
    use crate::ast;
    pub use ast::{
        get_common_label_filters, optimize, push_down_filters, simplify_expression,
        trim_filters_by_group_modifier,
    };
}

pub mod prelude {
    use crate::ast;
    use crate::binaryop;
    use crate::common;
    use crate::functions;
    use crate::parser;

    pub use ast::*;
    pub use binaryop::*;
    pub use common::*;
    pub use functions::*;
    pub use parser::*;
}
