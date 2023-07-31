use std::str::FromStr;

use crate::ast::{BinaryExpr, Expr};
use crate::common::{
    GroupModifier, GroupModifierOp, JoinModifier, JoinModifierOp, Operator, StringExpr, ValueType,
};
use crate::functions::AggregateFunction;
use crate::parser::function::parse_func_expr;
use crate::parser::parse_error::unexpected;
use crate::parser::tokens::Token;
use crate::parser::{
    extract_string_value, parse_number, unescape_ident, ParseError, ParseResult, Parser,
};

use super::aggregation::parse_aggr_func_expr;
use super::rollup::parse_rollup_expr;
use super::selector::parse_metric_expr;
use super::with_expr::parse_with_expr;

pub(super) fn parse_number_expr(p: &mut Parser) -> ParseResult<Expr> {
    let value = p.parse_number()?;
    Ok(Expr::from(value))
}

pub(super) fn parse_duration_expr(p: &mut Parser) -> ParseResult<Expr> {
    let duration = p.parse_duration()?;
    Ok(Expr::Duration(duration))
}

pub(super) fn parse_single_expr(p: &mut Parser) -> ParseResult<Expr> {
    if p.at(&Token::With) {
        let with = parse_with_expr(p)?;
        return Ok(Expr::With(with));
    }
    let expr = parse_single_expr_without_rollup_suffix(p)?;
    if p.peek_kind().is_rollup_start() {
        let re = parse_rollup_expr(p, expr)?;
        return Ok(re);
    }
    Ok(expr)
}

pub(super) fn parse_expression(p: &mut Parser) -> ParseResult<Expr> {
    let mut left = parse_single_expr(p)?;
    loop {
        if p.at_end() {
            break;
        }
        let token = p.current_token()?;
        let mut op_token = token.kind;

        // Hack incoming:
        // there is some ambiguity because of how the lexer handles negative numbers. In other words
        // -25 is parsed as [-25] as opposed to [Operator(Minus), 25]. So for example `time()-1` is
        // parsed as [time(), -1]. So we need to check for this case here.
        let mut right_scalar: Option<Expr> = None;
        if token.kind == Token::Number {
            if let Ok(right) = parse_number(token.text) {
                if right < 0_f64 {
                    // we have something like `time()-1000`
                    right_scalar = Some(Expr::from(right.abs()));
                    op_token = Token::OpMinus;
                }
            }
        }

        if !op_token.is_operator() {
            return Ok(left);
        }

        let operator = Operator::try_from(op_token)?;

        p.bump();

        let mut is_bool = false;
        if right_scalar.is_none() && p.at(&Token::Bool) {
            if !operator.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {operator}");
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
                    let msg = format!("modifier {} cannot be applied to {operator}", token.text);
                    return Err(p.syntax_error(&msg));
                }
                let join = parse_join_modifier(p)?;
                join_modifier = Some(join);
            }
        }

        let right = if let Some(right) = right_scalar {
            right
        } else {
            parse_single_expr(p)?
        };

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
            keep_metric_names,
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
    if let Expr::BinaryOperator(lhs_be) = &lhs {
        let precedence = lhs_be.op.precedence() as i16 - op.precedence() as i16;
        if (precedence < 0) || (precedence == 0 && op.is_right_associative()) {
            let right = lhs_be.right.as_ref().clone();
            let balanced = balance(
                right,
                op,
                rhs,
                group_modifier,
                join_modifier,
                return_bool,
                keep_metric_names,
            )?;

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

    // validate_scalar_op(&lhs, &rhs, op, return_bool)?;

    let expr = BinaryExpr {
        op,
        left: Box::new(lhs),
        right: Box::new(rhs),
        join_modifier,
        group_modifier,
        bool_modifier: return_bool,
        modifier: None,
        keep_metric_names,
    };

    Ok(Expr::BinaryOperator(expr))
}

fn balance_binary_op(be: BinaryExpr) -> Expr {
    let rp = be.op.precedence();

    // the duplicate match seems convoluted, but saves some cloning when we don't need to
    // balance
    let lp = match &be.left.as_ref() {
        Expr::BinaryOperator(bel) => Some(bel.op.precedence()),
        _ => None,
    };

    if let Some(lp) = lp {
        if rp < lp {
            return Expr::BinaryOperator(be);
        }

        if rp == lp && !be.op.is_right_associative() {
            return Expr::BinaryOperator(be);
        }

        let mut be = be;
        match be.left.as_mut() {
            Expr::BinaryOperator(bel) => {
                let mut be_left = std::mem::take(bel);
                be.left = std::mem::take(&mut bel.right);
                be_left.right = Box::new(balance_binary_op(be));
                return Expr::BinaryOperator(be_left);
            }
            _ => unreachable!("binary expr op changed type"),
        }
    }

    Expr::BinaryOperator(be)
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
            let value = Expr::string_literal(&extracted);
            p.bump();
            Ok(value)
        }
        Identifier => parse_ident_expr(p),
        Number => parse_number_expr(p),
        LeftParen => p.parse_parens_expr(),
        LeftBrace => parse_metric_expr(p),
        Duration => parse_duration_expr(p),
        OpPlus => parse_unary_plus_expr(p),
        OpMinus => parse_unary_minus_expr(p),
        _ => Err(unexpected(
            "",
            &tok.kind.to_string(),
            "Expr",
            Some(&tok.span),
        )),
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
            let msg = format!("unary Expr only allowed on expressions of type scalar or instant vector, got {:?}", t);
            Err(p.syntax_error(msg))
        }
    }
     */
    Ok(expr)
}

fn parse_unary_minus_expr(p: &mut Parser) -> ParseResult<Expr> {
    use ValueType::*;
    // assert(p.at(TokenKind::Minus)
    let span = p.last_token_range().unwrap();
    p.bump();
    let expr = parse_single_expr(p)?;

    let rt = expr.return_type();
    if !matches!(rt, InstantVector | Scalar) {
        let msg = format!(
            "unary Expr only allowed on expressions of type scalar or instant vector, got {:?}",
            rt
        );
        return Err(unexpected("", &rt.to_string(), &msg, Some(&span)));
    }

    // Substitute `-expr` with `0 - expr`
    let lhs = Expr::from(0.0);
    let binop_expr = BinaryExpr::new(Operator::Sub, lhs, expr);
    Ok(Expr::BinaryOperator(binop_expr))
}

pub(super) fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    let str = p.parse_string_expression()?;
    // todo: make sure
    Ok(str)
}

pub(super) fn handle_escape_ident(ident: &str) -> String {
    if ident.contains('\\') {
        unescape_ident(ident)
    } else {
        ident.to_string()
    }
}

/// parses expressions starting with `identifier` token.
fn parse_ident_expr(p: &mut Parser) -> ParseResult<Expr> {
    use Token::*;

    fn handle_metric_expression(p: &mut Parser) -> ParseResult<Expr> {
        p.back();
        parse_metric_expr(p)
    }

    let name = p.expect_identifier()?;

    // Look into the next token in order to determine how to parse
    // the current Expr.
    let kind = p.peek_kind();
    match kind {
        Eof | Offset => return handle_metric_expression(p),
        By | Without | LeftParen => {
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
        LeftBrace | LeftBracket | RightParen | Comma | At | KeepMetricNames => {
            return handle_metric_expression(p);
        }
        _ => {
            if kind.is_operator() {
                return handle_metric_expression(p);
            }
            // todo: check if we're parsing WITH
        }
    }

    let msg = format!("expecting identifier, found \"{}\"", &kind.to_string());
    Err(p.syntax_error(&msg))
}

fn is_aggr_func(name: &str) -> bool {
    AggregateFunction::from_str(name).is_ok()
}
