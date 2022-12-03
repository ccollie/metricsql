use std::str::FromStr;
use crate::ast::{BinaryExpr, DurationExpr, Expr};
use crate::common::{GroupModifier, GroupModifierOp, JoinModifier, JoinModifierOp, Operator, StringExpr, ValueType};
use crate::functions::AggregateFunction;
use crate::parser::{
    extract_string_value,
    parse_duration_value,
    parse_number as parse_number_base,
    ParseError,
    Parser,
    ParseResult,
    unescape_ident
};
use crate::parser::function::parse_func_expr;
use crate::parser::parse_error::unexpected;
use crate::parser::tokens::Token;
use crate::prelude::ParensExpr;

use super::aggregation::parse_aggr_func_expr;
use super::rollup::parse_rollup_expr;
use super::selector::parse_metric_expr;
use super::with_expr::parse_with_expr;

pub(super) fn parse_number(p: &mut Parser) -> ParseResult<f64> {
    let token = p.current_token()?;
    let value = parse_number_base(token.text)
            .map_err(|_|
                unexpected("Expr", token.text,"number", Some(&token.span))
            )?;

    p.bump();

    Ok(value)
}

pub(super) fn parse_number_expr(p: &mut Parser) -> ParseResult<Expr> {
    let value = parse_number(p)?;
    Ok(Expr::from(value))
}

pub(super) fn parse_duration(p: &mut Parser) -> ParseResult<DurationExpr> {
    use Token::*;

    let mut requires_step = false;
    let token = p.expect_one_of(&[Number, Duration])?;
    let last_ch: char = token.text.chars().last().unwrap();

    let value = match token.kind {
        Number => {
            // there is a bit of ambiguity between a Number with a unit and a Duration in the
            // case of tokens with the suffix 'm'. For example, does 60m mean 60 minutes or
            // 60 million. We accept Duration here to deal with that special case
            if last_ch == 'm' || last_ch == 'M' {
                // treat as minute
                parse_duration_value(token.text, 1)?
            } else {
                parse_number_base(token.text)? as i64
            }
        }
        Duration => {
            requires_step = last_ch == 'i' || last_ch == 'I';
            parse_duration_value(token.text, 1)?
        },
        _ => unreachable!("parse_duration"),
    };

    Ok(DurationExpr {
        value,
        requires_step,
    })
}

pub(super) fn parse_duration_expr(p: &mut Parser) -> ParseResult<Expr> {
    let duration = parse_duration(p)?;
    Ok(Expr::Duration(duration))
}

pub(super) fn parse_single_expr(p: &mut Parser) -> ParseResult<Expr> {
    if p.at(&Token::With) {
        let with = parse_with_expr(p)?;
        return Ok(Expr::With(with));
    }
    let e = parse_single_expr_without_rollup_suffix(p)?;
    if p.peek_kind().is_rollup_start() {
        let re = parse_rollup_expr(p, e)?;
        return Ok(re);
    }
    Ok(e)
}

pub(super) fn parse_expression(p: &mut Parser) -> ParseResult<Expr> {
    let mut left = parse_single_expr(p)?;
    loop {
        if p.at_end() {
            break;
        }
        let token = p.current_token()?;
        if !token.kind.is_operator() {
            return Ok(left);
        }

        let operator = Operator::try_from(token.kind)?;

        p.bump();

        let mut is_bool = false;
        if p.at(&Token::Bool) {
            if !operator.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {}", operator);
                return Err(p.syntax_error(&msg));
            }
            is_bool = true;
            p.bump();
        }

        let mut group_modifier: Option<GroupModifier> = None;
        let mut join_modifier: Option<JoinModifier> = None;

        if p.at_set(&[Token::On, Token::Ignoring]) {
            group_modifier = Some(parse_group_modifier(p)?);
            // join modifier
            let token = p.current_token()?;
            if [Token::GroupLeft, Token::GroupRight].contains(&token.kind) {
                if operator.is_set_operator() {
                    let msg = format!("modifier {} cannot be applied to {}", token.text, operator);
                    return Err(p.syntax_error(&msg));
                }
                let join = parse_join_modifier(p)?;
                join_modifier = Some(join);
            }
        }

        let right = parse_single_expr(p)?;

        let mut keep_metric_names = false;
        if p.at(&Token::KeepMetricNames) {
            p.bump();
            keep_metric_names = true;
        }

        left = balance(
            left,
            operator,
            right,
            group_modifier,
            join_modifier,
            is_bool,
            keep_metric_names
        )?;

    }

    Ok(left)
}

// see https://github.com/influxdata/promql/blob/eb8f592be73d3164ad7a723b9f3d6a7f565ca780/parse.go#L425
fn balance(
    lhs: Expr,
    op: Operator,
    rhs: Expr,
    group_modifier: Option<GroupModifier>,
    join_modifier: Option<JoinModifier>,
    return_bool: bool,
    keep_metric_names: bool,
) -> ParseResult<Expr> {

    match &lhs {
        Expr::BinaryOperator(lhs_be) => {
            let precedence = lhs_be.op.precedence() as i16 - op.precedence() as i16;
            if (precedence < 0) || (precedence == 0 && op.is_right_associative()) {
                let right = lhs_be.right.as_ref().clone();
                let balanced = balance(right, op, rhs, group_modifier, join_modifier, return_bool, keep_metric_names)?;

                // validate_scalar_op(&lhs_be.left,
                //                    &balanced,
                //                    lhs_be.op,
                //                    lhs_be.bool_modifier)?;

                let expr = BinaryExpr {
                    op: lhs_be.op,
                    left: lhs_be.left.clone(),
                    right: Box::new(balanced),
                    join_modifier: lhs_be.join_modifier.clone(),
                    group_modifier: lhs_be.group_modifier.clone(),
                    bool_modifier: lhs_be.bool_modifier,
                    modifier: None,
                    keep_metric_names: lhs_be.keep_metric_names,
                };
                return Ok(Expr::BinaryOperator(expr));
            }
        }
        _ => {}
    }

    // validate_scalar_op(&lhs, &rhs, op, return_bool)?;

    let expr = BinaryExpr {
        op,
        left: Box::new(lhs),
        right: Box::new(rhs),
        join_modifier,
        group_modifier,
        bool_modifier: return_bool,
        modifier: None,
        keep_metric_names
    };

    Ok(Expr::BinaryOperator(expr))
}

fn parse_group_modifier(p: &mut Parser) -> Result<GroupModifier, ParseError> {
    let tok = p.expect_one_of(&[Token::Ignoring, Token::On])?;

    let op: GroupModifierOp = match tok.kind {
        Token::Ignoring => GroupModifierOp::Ignoring,
        Token::On => GroupModifierOp::On,
        _ => unreachable!(),
    };

    let mut args = p.parse_ident_list()?;
    args.sort();

    Ok(GroupModifier::new(op, args))
}

fn parse_join_modifier(p: &mut Parser) -> ParseResult<JoinModifier> {
    let tok = p.expect_one_of(&[Token::GroupLeft, Token::GroupRight])?;

    let op = match tok.kind {
        Token::GroupLeft => JoinModifierOp::GroupLeft,
        Token::GroupRight => JoinModifierOp::GroupRight,
        _ => unreachable!(),
    };

    let mut labels = if !p.at(&Token::LeftParen) {
        // join modifier may ignore ident list.
        vec![]
    } else {
        p.parse_ident_list()?
    };

    labels.sort();
    Ok(JoinModifier::new(op, labels))
}

pub(super) fn parse_single_expr_without_rollup_suffix(p: &mut Parser) -> ParseResult<Expr> {
    use Token::*;

    let tok = p.current_token()?;
    match &tok.kind {
        StringLiteral => {
            let extracted = extract_string_value(tok.text)?;
            let value = Expr::string_literal(&*extracted);
            p.bump();
            Ok(value)
        },
        Identifier => parse_ident_expr(p),
        Number => parse_number_expr(p),
        LeftParen => parse_parens_expr(p),
        LeftBrace => parse_metric_expr(p),
        Duration => parse_duration_expr(p),
        OpPlus => parse_unary_plus_expr(p),
        OpMinus => parse_unary_minus_expr(p),
        _ => Err(unexpected("", &tok.kind.to_string(), "Expr", Some(&tok.span))),
    }
}

/// returns positive duration in milliseconds for the given s
/// and the given step.
///
/// Duration in s may be combined, i.e. 2h5m or 2h-5m.
///
/// Error is returned if the duration in s is negative.
pub(super) fn parse_positive_duration(p: &mut Parser) -> ParseResult<DurationExpr> {
    // Verify the duration in seconds without explicit suffix.
    let duration = parse_duration(p)?;
    let val = duration.value(1);
    if val < 0 {
        Err(ParseError::InvalidDuration(duration.to_string()))
    } else {
        Ok(duration)
    }
}

fn parse_unary_plus_expr(p: &mut Parser) -> ParseResult<Expr> {
    p.expect(&Token::OpPlus)?;
    let expr = parse_single_expr(p)?;
    /*
    let t = checkAST(p, &expr)?;
    match t {
        ReturnType::Scalar | ReturnType::InstantVector => Ok(expr),
        _ => {
            let msg = format!("unary Expr only allowed on Exprs of type scalar or instant vector, got {:?}", t);
            Err(p.syntax_error(msg))
        }
    }
     */
    Ok(expr)
}

fn parse_unary_minus_expr(p: &mut Parser) -> ParseResult<Expr> {
    // assert(p.at(TokenKind::Minus)
    let span = p.last_token_range().unwrap();
    p.bump();
    let expr = parse_single_expr(p)?;

    let rt = expr.return_type();
    match rt {
        ValueType::InstantVector | ValueType::Scalar => {}
        _ => {
            let msg = format!(
                "unary Expr only allowed on Exprs of type scalar or instant vector"
            );
            return Err(unexpected( "", &rt.to_string(),&msg, Some(&span)));
        }
    }

    // Substitute `-expr` with `0 - expr`
    let lhs = Expr::from(0.0);
    let binop_expr = BinaryExpr::new(Operator::Sub, lhs, expr);
    Ok(Expr::BinaryOperator(binop_expr))
}

pub(super) fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    let str = parse_string_expression(p)?;
    // todo: make sure
    Ok(str)
}

pub(super) fn parse_string_expression(p: &mut Parser) -> ParseResult<StringExpr> {
    use Token::*;

    let mut tok = p.current_token()?;
    let mut result = StringExpr::default();

    loop {

        match &tok.kind {
            StringLiteral => {
                let value = extract_string_value(tok.text)?;
                if !value.is_empty() {
                    result.push_str(&value);
                }
            },
            Identifier => {
                // clone to avoid borrow issues later
                let ident = handle_escape_ident(tok.text);
                if let Some(wa) = p.lookup_with_expr(&ident) {
                    match &wa.expr {
                        Expr::StringExpr(se) => {
                            for segment in se.iter() {
                                result.push(segment)
                            }
                        }
                        _ => {
                            // we'll resolve later
                            result.push_ident(&ident)
                        }
                    }
                } else {
                    result.push_ident(&ident)
                }
            }
            _ => {
                return Err(p.token_error(&[StringLiteral, Identifier]));
            }
        }

        if let Some(token) = p.next_token() {
            tok = token;
        } else {
            break;
        }

        if tok.kind != OpPlus {
            break;
        }

        // if p.at_end() {
        //     break;
        // }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.
        if let Some(token) = p.next_token() {
            tok = token;
        } else {
            break;
        }

        if tok.kind == StringLiteral {
            // "s1" + "s2"
            continue;
        }

        if tok.kind != Identifier {
            // "s" + unknownToken
            p.back();
            break;
        }

        // Look after ident
        match p.next_token() {
            None => break,
            Some(t) => {
                tok = t;
                if tok.kind == LeftParen || tok.kind == LeftBrace {
                    p.back();
                    p.back();
                    // `"s" + m(` or `"s" + m{`
                    break;
                }
            }
        }

        // "s" + ident
        tok = p.prev_token().unwrap();
    }

    Ok(result)
}

pub(super) fn parse_parens_expr(p: &mut Parser) -> ParseResult<Expr> {
    let list = parse_arg_list(p)?;
    Ok(Expr::Parens(ParensExpr::new(list)))
}

pub(super) fn parse_arg_list(p: &mut Parser) -> ParseResult<Vec<Expr>> {
    use Token::*;
    p.expect(&LeftParen)?;
    p.parse_comma_separated(&[RightParen], parse_expression)
}

pub(super) fn handle_escape_ident(ident: &str) -> String {
    if ident.contains(r#"\"#) {
        unescape_ident(ident)
    } else {
        ident.to_string()
    }
}

/// parses expressions starting with `identifier` token.
fn parse_ident_expr(p: &mut Parser) -> ParseResult<Expr> {
    use Token::*;

    let name = p.expect_identifier()?;

    // Look into the next token in order to determine how to parse
    // the current Expr.
    if p.at_end() {
        p.back();
        return parse_metric_expr(p);
    }
    let kind = p.peek_kind();
    match kind {
        Eof | Offset => {
            p.back();
            return parse_metric_expr(p);
        }
        By | Without | LeftParen => {
            // ugly: avoid borrow problems
            let is_left_paren = kind == LeftParen;
            p.back();
            if is_aggr_func(&name) {
                return parse_aggr_func_expr(p);
            }
            if is_left_paren {
                return parse_func_expr(p);
            }
            return parse_metric_expr(p);
        }
        LeftBrace | LeftBracket | RightParen | Comma | At => {
            p.back();
            return parse_metric_expr(p);
        }
        _ => {
            if kind.is_operator() {
                p.back();
                return parse_metric_expr(p);
            }
            // todo: check if we're parsing WITH
        }
    }

    Err(unexpected( "", &kind.to_string(),"identifier", None))
}

fn is_aggr_func(name: &str) -> bool {
    AggregateFunction::from_str(name).is_ok()
}
