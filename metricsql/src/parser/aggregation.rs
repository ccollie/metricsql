use std::str::FromStr;
use crate::ast::{AggregateModifier, AggregateModifierOp, AggrFuncExpr, Expression};
use crate::functions::{AggregateFunction, BuiltinFunction};
use crate::lexer::TokenKind;
use crate::parser::{ParseErr, ParseError, Parser, ParseResult};
use super::expr::{parse_arg_list, parse_number};
use super::function::validate_args;


/// parse_aggr_func_expr parses an aggregation expression.
///
///		<aggr_op> (<Vector_expr>) [by|without <labels>] [limit <number>]
///		<aggr_op> [by|without <labels>] (<Vector_expr>) [limit <number>]
///
pub(super) fn parse_aggr_func_expr(p: &mut Parser) -> ParseResult<Expression> {
    let tok = p.current_token()?;
    let mut span = tok.span;

    let func: AggregateFunction = AggregateFunction::from_str(tok.text)?;

    fn handle_prefix(p: &mut Parser, ae: &mut AggrFuncExpr) -> ParseResult<()> {
        ae.modifier = Some(parse_aggregate_modifier(p)?);
        handle_args(p, ae)
    }

    fn handle_args(p: &mut Parser, ae: &mut AggrFuncExpr) -> ParseResult<()> {
        ae.args = parse_arg_list(p)?;

        validate_args(&BuiltinFunction::Aggregate(ae.function), &ae.args)?;
        let kind = p.peek_kind();
        // Verify whether func suffix exists.
        if ae.modifier.is_none() && kind.is_aggregate_modifier() {
            ae.modifier = Some(parse_aggregate_modifier(p)?);
        }

        if p.at(TokenKind::Limit) {
            ae.limit = parse_limit(p)?;
        }

        Ok(())
    }

    let mut ae: AggrFuncExpr = AggrFuncExpr::new(&func);
    p.bump();

    let kind = p.peek_kind();
    if kind.is_aggregate_modifier() {
        handle_prefix(p, &mut ae)?;
    } else if kind == TokenKind::LeftParen {
        handle_args(p, &mut ae)?;
    } else {
        return Err(p.token_error(&[
            TokenKind::By,
            TokenKind::Without,
            TokenKind::LeftParen
        ]));
    }

    p.update_span(&mut span);
    ae.span = span;

    Ok(Expression::Aggregation(ae))
}

fn parse_aggregate_modifier(p: &mut Parser) -> ParseResult<AggregateModifier> {
    let tok = p.expect_one_of(&[TokenKind::By, TokenKind::Without])?;
    let op = match tok.kind {
        TokenKind::By => AggregateModifierOp::By,
        TokenKind::Without => AggregateModifierOp::Without,
        _ => unreachable!()
    };

    let args = p.parse_ident_list()?;
    let res = AggregateModifier::new(op, args);

    Ok(res)
}

fn parse_limit(p: &mut Parser) -> ParseResult<usize> {
    p.expect(TokenKind::Limit)?;
    let v = parse_number(p)?;
    if v < 0.0 || !v.is_finite() {
        // invalid value
        let msg = format!("LIMIT should be a positive integer. Found {} ", v).to_string();
        let err = ParseErr::new(&msg, p.input, p.last_token_range().unwrap());
        return Err(ParseError::Unexpected(err));
    }
    Ok(v as usize)
}