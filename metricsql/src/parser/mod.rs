use once_cell::sync::OnceCell;

use crate::ast::{Expr, WithArgExpr};
use crate::parser::expr::{parse_expression};
use crate::parser::with_expr::{check_duplicate_with_arg_names, must_parse_with_arg_expr};

pub use duration::parse_duration_value;
pub use function::validate_function_args;
pub use number::{get_number_suffix, parse_number};
pub use parse_error::*;
pub use parser::*;
pub use regexp_cache::{compile_regexp, is_empty_regex};
pub use selector::parse_metric_expr;
pub use utils::{escape_ident, extract_string_value, quote, unescape_ident};

mod aggregation;
mod expr;
mod function;
mod regexp_cache;
mod rollup;
mod selector;
mod with_expr;

pub mod parse_error;
pub mod parser;
pub mod number;
pub mod duration;
pub mod utils;
pub mod tokens;

// tests
mod expand_expr;
#[cfg(test)]
mod expand_with_test;
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;
#[cfg(test)]
mod lexer_tests;

pub fn parse(input: &str) -> ParseResult<Expr> {
    let mut parser = Parser::new(input)?;
    let expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    parser.expand_if_needed(expr)
}

/// Expands WITH expressions inside q and returns the resulting
/// PromQL without WITH expressions.
pub fn expand_with_exprs(q: &str) -> Result<String, ParseError> {
    let e = parse(q)?;
    Ok(format!("{}", e))
}

static DEFAULT_EXPRS: [&str; 1] = [
    // ttf - time to fuckup
    "ttf(freev) = smooth_exponential(
        clamp_max(clamp_max(-freev, 0) / clamp_max(deriv_fast(freev), 0), 365*24*3600),
        clamp_max(step()/300, 1)
    )",
];

fn get_default_with_arg_exprs() -> &'static Vec<WithArgExpr> {
    static INSTANCE: OnceCell<Vec<WithArgExpr>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let was: [WithArgExpr; 1] = DEFAULT_EXPRS.map(|expr| {
            let res = must_parse_with_arg_expr(expr);
            res.unwrap()
        });

        if let Err(err) = check_duplicate_with_arg_names(&was.to_vec()) {
            panic!("BUG: {:?}", err)
        }
        was.to_vec()
    })
}
