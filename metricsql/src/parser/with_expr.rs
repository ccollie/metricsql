use crate::ast::{WithArgExpr, WithExpr};
use crate::parser::parse_expression;
use crate::parser::{ParseError, ParseResult, Parser};
use crate::parser::tokens::Token;
use std::collections::HashSet;
use crate::parser::parse_error::unexpected;


/// parses `WITH (withArgExpr...) expr`.
pub(super) fn parse_with_expr(p: &mut Parser) -> ParseResult<WithExpr> {
    use Token::*;

    p.expect(&With)?;
    p.expect(&LeftParen)?;

    p.template_parsing_depth += 1;
    push_stack(p);

    loop {
        if p.at(&RightParen) {
            p.template_parsing_depth -= 1;
            p.bump();
            break;
        }
        let item = parse_with_arg_expr(p)?;
        store_with_expr(p, item);

        let tok = p.current_token()?;
        match tok.kind {
            Comma => {
                p.bump();
                continue;
            }
            RightParen => {
                p.template_parsing_depth -= 1;
                p.bump();
                break;
            }
            _ => {
                return Err(
                    unexpected(
                        "with(expr...) expression",
                        &tok.kind.to_string(),
                        ") or ,",
                        Some(&tok.span)
                    )
                );
            }
        }
    }

    if let Some(was) = p.with_stack.last() {
        // end:
        check_duplicate_with_arg_names(was)?;
    }

    let expr = parse_expression(p)?;

    let args = pop_stack(p);

    Ok(WithExpr::new(expr, args))
}

fn parse_with_arg_expr(p: &mut Parser) -> ParseResult<WithArgExpr> {
    use Token::*;
    let name = p.expect_identifier()?;

    let is_function: bool;

    let args = if p.at(&LeftParen) {
        is_function = true;
        // Parse func args.

        // push_stack(p)
        // Make sure all the args have different names
        let mut m: HashSet<String> = HashSet::with_capacity(6);

        p.expect(&LeftParen)?;
        let args = p.parse_comma_separated(&[RightParen], |parser| {
            let ident = parser.expect_identifier()?;
            if m.contains(&ident) {
                // todo: syntax_error()?
                let msg = format!("withArgExpr: duplicate arg name: {}", ident);
                return Err(ParseError::General(msg));
            } else {
                m.insert(ident.clone());
            }
            Ok(ident)
        })?;

        args
    } else {
        is_function = false;
        vec![]
    };

    p.expect(&Equal)?;

    match parse_expression(p) {
        Ok(expr) => {
            if is_function {
                Ok(WithArgExpr::new_function(name.to_string(), expr, args))
            } else {
                Ok(WithArgExpr::new(name.to_string(), expr))
            }
        }
        Err(e) => {
            let msg = format!("withArgExpr: cannot parse expression for {}: {:?}", name, e);
            return Err(ParseError::General(msg));
        }
    }
}

pub(super) fn must_parse_with_arg_expr(s: &str) -> ParseResult<WithArgExpr> {
    let mut p = Parser::new(s)?;
    if p.is_eof() {
        return Err(ParseError::UnexpectedEOF);
    }
    let expr = parse_with_arg_expr(&mut p)?;
    if !p.is_eof() {
        let msg = format!("BUG: cannot parse {}: unparsed data", s);
        return Err(ParseError::SyntaxError(msg));
    }
    Ok(expr)
}

pub(super) fn check_duplicate_with_arg_names(was: &Vec<WithArgExpr>) -> ParseResult<()> {
    let mut m: HashSet<String> = HashSet::with_capacity(was.len());

    for wa in was {
        if m.contains(&wa.name) {
            return Err(ParseError::DuplicateArgument(wa.name.clone()));
        }
        m.insert(wa.name.clone());
    }
    Ok(())
}

pub(super) fn push_stack(p: &mut Parser) {
    p.with_stack.push(Vec::with_capacity(4));
}

pub(super) fn pop_stack(p: &mut Parser) -> Vec<WithArgExpr> {
    p.with_stack.pop().unwrap_or_default()
}

fn store_with_expr(p: &mut Parser, expr: WithArgExpr) {
    match p.with_stack.last_mut() {
        Some(top) => top.push(expr),
        None => {
            panic!("Bug: trying to fetch from an empty WITH stack")
        }
    }
}
