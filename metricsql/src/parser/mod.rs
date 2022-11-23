pub use duration::parse_duration_value;
pub use function::validate_function_args;
pub use number::{get_number_suffix, parse_number};
pub use parse_error::*;
pub use parser::*;
pub use regexp_cache::{compile_regexp, is_empty_regex};
pub use selector::parse_metric_expr;
pub use utils::{escape_ident, extract_string_value, quote, unescape_ident};

use crate::ast::Expr;
use crate::parser::expr::parse_expression;
use crate::prelude::check_ast;

mod aggregation;
mod expand;
mod expr;
mod function;
mod regexp_cache;
mod rollup;
mod selector;
mod with_expr;

pub mod duration;
pub mod number;
pub mod parse_error;
pub mod parser;
pub mod symbol_provider;
pub mod tokens;
pub mod utils;

// tests
#[cfg(test)]
mod expand_with_test;
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;

pub fn parse(input: &str) -> ParseResult<Expr> {
    let mut parser = Parser::new(input)?;
    let expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    let expr = parser.expand_if_needed(expr)?;
    check_ast(expr).map_err(|err| ParseError::General(err.to_string()))
}

/// Expands WITH expressions inside q and returns the resulting
/// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, ParseError> {
    let e = parse(q)?;
    Ok(format!("{}", e))
}
