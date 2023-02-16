use crate::ast::{WithArgExpr, WithExpr};
use crate::lexer::{unescape_ident, TokenKind};
use crate::parser::expr::parse_expression;
use crate::parser::parser::unexpected;
use crate::parser::{ParseError, ParseResult, Parser};
use std::collections::HashSet;

/// parses `WITH (withArgExpr...) expr`.
pub(super) fn parse_with_expr(p: &mut Parser) -> ParseResult<WithExpr> {
    use TokenKind::*;

    p.expect(With)?;
    p.expect(LeftParen)?;

    p.template_parsing_depth += 1;
    push_stack(p);

    loop {
        if p.at(RightParen) {
            p.template_parsing_depth -= 1;
            p.bump();
            break;
        }
        let item = parse_with_arg_expr(p)?;
        store_with_expr(p, item);

        match p.peek_kind() {
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
                return Err(unexpected(p, "parse_with_expr", ") or ,", None));
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
    let tok = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(tok.text);
    let is_function: bool;

    let args = if p.at(TokenKind::LeftParen) {
        is_function = true;
        // Parse func args.

        // push_stack(p)
        let args = p.parse_ident_list()?;

        // Make sure all the args have different names
        let mut m: HashSet<String> = HashSet::with_capacity(4);
        for arg in args.iter() {
            if m.contains(arg) {
                let msg = format!("withArgExpr: duplicate arg name: {}", arg);
                return Err(ParseError::General(msg));
            }
            m.insert(arg.clone());
        }

        args
    } else {
        is_function = false;
        vec![]
    };

    p.expect(TokenKind::Equal)?;

    match parse_expression(p) {
        Ok(expr) => {
            if is_function {
                Ok(WithArgExpr::new_function(name, expr, args))
            } else {
                Ok(WithArgExpr::new(name, expr))
            }
        }
        Err(e) => {
            let msg = format!("withArgExpr: cannot parse expression for {}: {:?}", name, e);
            return Err(ParseError::General(msg));
        }
    }
}

pub(super) fn must_parse_with_arg_expr(s: &str) -> ParseResult<WithArgExpr> {
    let mut p = Parser::new(s);
    let tok = p.peek_kind();
    if tok == TokenKind::Eof {
        return Err(ParseError::UnexpectedEOF);
    }
    let expr = parse_with_arg_expr(&mut p)?;
    if !p.is_eof() {
        let msg = format!("BUG: cannot parse {}: unparsed data", s);
        return Err(ParseError::General(msg));
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
