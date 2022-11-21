use std::collections::HashSet;
use crate::ast::{WithArgExpr, WithExpr};
use crate::lexer::{TokenKind, unescape_ident};
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::expr::parse_expression;

/// parses `WITH (withArgExpr...) expr`.
pub(super) fn parse_with_expr(p: &mut Parser) -> ParseResult<WithExpr> {
    let mut span = p.last_token_range().unwrap();

    p.expect(TokenKind::With)?;
    p.expect(TokenKind::LeftParen)?;

    let was_in_with = p.parsing_with;
    p.parsing_with = true;

    let was = p.parse_comma_separated(&[TokenKind::RightParen],
                                      parse_with_arg_expr)?;

    if !was_in_with {
        p.parsing_with = false
    }

    // end:
    check_duplicate_with_arg_names(&was)?;

    let expr = parse_expression(p)?;
    p.update_span(&mut span);
    Ok(WithExpr::new(expr, was, span))
}

fn parse_with_arg_expr(p: &mut Parser) -> ParseResult<WithArgExpr> {

    let tok = p.expect_token(TokenKind::Ident)?;
    let name = unescape_ident(tok.text);

    let args = if p.at(TokenKind::LeftParen) {
        // Parse func args.
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
        vec![]
    };

    p.expect(TokenKind::Equal)?;
    let expr = match parse_expression(p) {
        Ok(e) => e,
        Err(e) => {
            let msg = format!("withArgExpr: cannot parse expression for {}: {:?}", name, e);
            return Err(ParseError::General(msg));
        }
    };

    Ok(WithArgExpr::new(name, expr, args))
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

pub(super) fn check_duplicate_with_arg_names(was: &[WithArgExpr]) -> ParseResult<()> {
    let mut m: HashSet<String> = HashSet::with_capacity(was.len());

    for wa in was {
        if m.contains(&wa.name) {
            return Err(ParseError::DuplicateArgument(wa.name.clone()));
        }
        m.insert(wa.name.clone());
    }
    Ok(())
}