use once_cell::sync::OnceCell;
pub use parse_error::*;
pub use parser::*;
pub use regexp_cache::compile_regexp;
use crate::ast::{Expression, WithArgExpr};
use crate::lexer::TokenKind;
use crate::parser::expand_with::expand_with_expr;
use crate::parser::expr::parse_expression;
use crate::parser::simplify::simplify_expr;
use crate::parser::with_expr::{check_duplicate_with_arg_names, must_parse_with_arg_expr};

mod aggregation;
mod expand_with;
mod expr;
mod function;
mod regexp_cache;
mod rollup;
mod selector;
mod simplify;
mod with_expr;

pub mod parse_error;
pub mod parser;

// tests
#[cfg(test)]
mod parser_example_test;
#[cfg(test)]
mod parser_test;
#[cfg(test)]
mod expand_with_test;


pub fn parse(input: &str) -> ParseResult<Expression> {
    let mut parser = Parser::new(input);
    let tok = parser.peek_kind();
    if tok == TokenKind::Eof {
        let msg = format!("cannot parse the first token {}", input);
        return Err(ParseError::General(msg));
    }
    let expr = parse_expression(&mut parser)?;
    if !parser.is_eof() {
        let msg = "unparsed data".to_string();
        return Err(ParseError::General(msg));
    }
    let was = get_default_with_arg_exprs();
    match expand_with_expr(was, &expr) {
        Ok(expr) => simplify_expr(&expr),
        Err(e) => Err(e),
    }
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

fn get_default_with_arg_exprs() -> &'static [WithArgExpr; 1] {
    static INSTANCE: OnceCell<[WithArgExpr; 1]> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let was: [WithArgExpr; 1] =
            DEFAULT_EXPRS.map(|expr| {
                let res = must_parse_with_arg_expr(expr);
                res.unwrap()
            });

        if let Err(err) = check_duplicate_with_arg_names(&was) {
            panic!("BUG: {:?}", err)
        }
        was
    })
}





