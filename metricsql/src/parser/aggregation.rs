use super::expr::{parse_arg_list, parse_number};
use crate::ast::{AggregationExpr, Expr};
use crate::common::{AggregateModifier, AggregateModifierOp};
use crate::functions::{AggregateFunction, BuiltinFunction};
use crate::parser::{ParseErr, ParseError, ParseResult, Parser};
use std::str::FromStr;
use crate::parser::function::validate_function_args;
use crate::parser::tokens::Token;

/// parse_aggr_func_expr parses an aggregation Expr.
///
///		<aggr_op> (<Vector_expr>) [by|without <labels>] [limit <number>]
///		<aggr_op> [by|without <labels>] (<Vector_expr>) [limit <number>]
///
pub(super) fn parse_aggr_func_expr(p: &mut Parser) -> ParseResult<Expr> {
    let tok = p.expect_identifier()?;

    let func = AggregateFunction::from_str(&tok)?;

    fn handle_prefix(p: &mut Parser, func: AggregateFunction) -> ParseResult<Expr> {
        let modifier = Some(parse_aggregate_modifier(p)?);
        handle_args(p, func, modifier)
    }

    fn handle_args(p: &mut Parser, func: AggregateFunction, modifier: Option<AggregateModifier>) -> ParseResult<Expr> {
        let args = parse_arg_list(p)?;

        validate_function_args(&BuiltinFunction::Aggregate(func), &args)?;
        let mut ae = AggregationExpr::new(&func, args);
        let kind = p.peek_kind();
        // Verify whether func suffix exists.
        ae.modifier = if modifier.is_none() && kind.is_aggregate_modifier() {
           Some(parse_aggregate_modifier(p)?)
        } else {
            modifier
        };

        if p.at(&Token::Limit) {
            ae.limit = parse_limit(p)?;
        }

        Ok(Expr::Aggregation(ae))
    }

    let kind = p.peek_kind();
    return if kind.is_aggregate_modifier() {
        handle_prefix(p, func)
    } else if kind == Token::LeftParen {
        handle_args(p, func, None)
    } else {
        Err(p.token_error(&[Token::By, Token::Without, Token::LeftParen]))
    }
}

fn parse_aggregate_modifier(p: &mut Parser) -> ParseResult<AggregateModifier> {
    let tok = p.expect_one_of(&[Token::By, Token::Without])?;
    let op = match tok.kind {
        Token::By => AggregateModifierOp::By,
        Token::Without => AggregateModifierOp::Without,
        _ => unreachable!(),
    };

    let args = p.parse_ident_list()?;
    let res = AggregateModifier::new(op, args);

    Ok(res)
}

fn parse_limit(p: &mut Parser) -> ParseResult<usize> {
    p.expect(&Token::Limit)?;
    let saved_pos = p.cursor;
    let v = parse_number(p)?;
    if v < 0.0 || !v.is_finite() {

        let end_pos = p.cursor;
        let span = saved_pos..end_pos;
        // invalid value
        let msg = format!("LIMIT should be a positive integer. Found {} ", v).to_string();
        let err = ParseErr::new(&msg, span);
        return Err(ParseError::Unexpected(err));
    }
    Ok(v as usize)
}
