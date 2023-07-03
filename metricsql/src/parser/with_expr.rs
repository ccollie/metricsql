use crate::ast::{WithArgExpr, WithExpr};
use crate::parser::tokens::Token;
use crate::parser::{parse_expression, syntax_error};
use crate::parser::{ParseError, ParseResult, Parser};
use std::collections::HashSet;

/// parses `WITH (withArgExpr...) expr`.
pub(super) fn parse_with_expr(p: &mut Parser) -> ParseResult<WithExpr> {
    use Token::*;

    p.expect(&With)?;
    p.expect(&LeftParen)?;

    p.needs_expansion = true;

    push_stack(p);

    loop {
        if p.at(&RightParen) {
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
                p.bump();
                break;
            }
            _ => {
                // force error
                p.expect_one_of(&[Comma, RightParen])?;
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

/// parses `var = expr` | `var(arg, ...) = expr`.
fn parse_with_arg_expr(p: &mut Parser) -> ParseResult<WithArgExpr> {
    use Token::*;
    let name = p.expect_identifier()?;

    let args = if p.at(&LeftParen) {
        // Parse func args.
        // push_stack(p)
        // Make sure all the args have different names
        let mut m: HashSet<String> = HashSet::with_capacity(6);

        p.expect(&LeftParen)?;
        p.parse_comma_separated(&[RightParen], move |parser| {
            let ident = parser.expect_identifier()?;
            if m.contains(&ident) {
                let msg = format!("withArgExpr: duplicate arg name: {ident}");
                return Err(parser.syntax_error(&msg));
            } else {
                m.insert(ident.clone());
            }
            Ok(ident)
        })?
    } else {
        vec![]
    };

    p.expect(&Equal)?;

    p.needs_expansion = true;

    let start = p.cursor;
    match parse_expression(p) {
        Ok(expr) => {
            let end = p.cursor - 1;
            let mut wae = WithArgExpr::new(name.to_string(), expr, args);
            wae.token_range = start..end;
            Ok(wae)
        }
        Err(e) => {
            let msg = format!("cannot parse expression for {}: {:?}", name, e);
            let end = p.cursor - 1;
            let range = start..end;
            let err = syntax_error(&msg, &range, "WithArgExpr".to_string());
            return Err(err);
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
