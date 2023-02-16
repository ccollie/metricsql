#![forbid(unsafe_code)]
extern crate core;
extern crate enquote;
extern crate logos;
extern crate once_cell;
extern crate phf;
extern crate regex;
extern crate serde;
extern crate thiserror;

pub mod ast;
pub mod binaryop;
pub mod error;
mod lexer;

pub mod utils {
    use crate::lexer;
    pub use lexer::{escape_ident, parse_number, quote};
}

pub mod common;
pub mod functions;
pub mod parser;
pub mod transform;

pub use lexer::TextSpan;

pub mod hir;

pub mod optimize {
    use crate::hir;
    pub use hir::{
        get_common_label_filters, optimize, push_down_filters, simplify_expression,
        trim_filters_by_group_modifier,
    };
}

pub mod prelude {
    use crate::ast;
    use crate::binaryop;
    use crate::common;
    use crate::functions;
    use crate::hir;
    use crate::lexer;
    use crate::parser;

    pub use ast::*;
    pub use binaryop::*;
    pub use common::*;
    pub use functions::*;
    pub use hir::*;
    pub use lexer::TextSpan;
    pub use parser::*;
}
