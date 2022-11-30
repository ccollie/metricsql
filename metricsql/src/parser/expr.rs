use std::str::FromStr;

use crate::ast::{
    BExpression,
    BinaryOp,
    BinaryOpExpr,
    DurationExpr,
    Expression,
    GroupModifier,
    GroupModifierOp,
    JoinModifier,
    JoinModifierOp,
    NumberExpr,
    ReturnValue,
    StringExpr,
    StringTokenType
};
use crate::ast::Expression::BinaryOperator;
use crate::functions::AggregateFunction;
use crate::lexer::{extract_string_value, get_number_suffix, parse_float, parse_number_with_unit, TokenKind};
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::parser::unexpected;
use crate::TextSpan;

use super::aggregation::parse_aggr_func_expr;
use super::function::parse_function;
use super::rollup::parse_rollup_expr;
use super::selector::parse_metric_expr;
use super::with_expr::parse_with_expr;


pub(super) fn parse_number(p: &mut Parser) -> ParseResult<f64> {
    use TokenKind::*;
    let token = p.current_token()?;
    let kind = token.kind;

    fn raise_error(p: &mut Parser, span: TextSpan) -> ParseResult<f64> {
        return Err(unexpected(p, "expression", "number", Some(span) ))
    }

    let value = match kind {
        Number => parse_float(token.text),
        NumberWithUnit => parse_number_with_unit(token.text),
        Duration => {
            // there is a bit of ambiguity between a NumberWithUnit and a Duration in the
            // case of tokens with the prefix 'm'. For example, does 60m mean 60 minutes or
            // 60 million. We accept Duration here to deal with that special case
            let suffix = get_number_suffix(token.text);
            if let Some(a_suffix) = suffix {
                if a_suffix.0 == "m" {
                    return parse_number_with_unit(token.text);
                }
            }
            return raise_error(p, token.span)
        }
        _ => {
            return raise_error(p, token.span)
        }
    };

    if value.is_ok() {
        p.bump();
    }

    value
}

pub(super) fn parse_number_expr(p: &mut Parser) -> ParseResult<Expression> {
    let span = p.last_token_range().unwrap();
    let value = parse_number(p)?;
    Ok(Expression::Number(NumberExpr::new(value, span)))
}

pub(super) fn parse_duration(p: &mut Parser) -> ParseResult<DurationExpr> {
    use TokenKind::*;
    let tok = p.expect_one_of(&[Number, Duration, NumberWithUnit])?;
    DurationExpr::new(tok.text, tok.span)
}

pub(super) fn parse_duration_expr(p: &mut Parser) -> ParseResult<Expression> {
    let duration = parse_duration(p)?;
    Ok(Expression::Duration(duration))
}

pub(super) fn parse_single_expr(p: &mut Parser) -> ParseResult<Expression> {
    if p.at(TokenKind::With) {
        let with = parse_with_expr(p)?;
        return Ok(Expression::With(with));
    }
    let e = parse_single_expr_without_rollup_suffix(p)?;
    if p.peek_kind().is_rollup_start() {
        let re = parse_rollup_expr(p, e)?;
        return Ok(re);
    }
    Ok(e)
}

pub(super) fn parse_expression(p: &mut Parser) -> ParseResult<Expression> {
    let mut left = parse_single_expr(p)?;
    loop {
        if p.at_end() {
            break;
        }
        let token = p.current_token()?;
        if !token.kind.is_operator() {
            return Ok(left);
        }

        let operator = BinaryOp::try_from(token.text)?;

        p.bump();

        let mut is_bool = false;
        if p.at(TokenKind::Bool) {
            if !operator.is_comparison() {
                let msg = format!("bool modifier cannot be applied to {}", operator);
                return Err(ParseError::General(msg));
            }
            is_bool = true;
            p.bump();
        }

        let mut group_modifier: Option<GroupModifier> = None;
        let mut join_modifier: Option<JoinModifier> = None;

        if p.at_set(&[TokenKind::On, TokenKind::Ignoring]) {
            group_modifier = Some(parse_group_modifier(p)?);
            // join modifier
            let token = p.current_token()?;
            if [TokenKind::GroupLeft, TokenKind::GroupRight].contains(&token.kind) {
                if operator.is_set_operator() {
                    let msg = format!("modifier {} cannot be applied to {}", token.text, operator);
                    return Err(ParseError::General(msg));
                }
                let join = parse_join_modifier(p)?;
                join_modifier = Some(join);
            }
        }

        let right = parse_single_expr(p)?;

        left = balance(left, operator, right, group_modifier, join_modifier, is_bool)?;
    }

    Ok(left)
}


// see https://github.com/influxdata/promql/blob/eb8f592be73d3164ad7a723b9f3d6a7f565ca780/parse.go#L425
fn balance(lhs: Expression,
           op: BinaryOp,
           rhs: Expression,
           group_modifier: Option<GroupModifier>,
           join_modifier: Option<JoinModifier>,
           return_bool: bool) -> ParseResult<Expression> {
    use ReturnValue::*;

    fn validate_scalar_op(
        left: &Expression,
        right: &Expression,
        op: BinaryOp,
        returns_bool: bool,
        // todo: span
    ) -> ParseResult<()> {
        match (left.return_value(), right.return_value()) {
            (Scalar, Scalar) => {
                if op.is_comparison() && !returns_bool {
                    // todo: better error, including position
                    return Err(
                        ParseError::General("comparisons between scalars must use BOOL modifier".to_string())
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }

    let span = lhs.span().cover(rhs.span());

    match &lhs {
        BinaryOperator(lhs_be) => {
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
                )?;

                // validate_scalar_op(&lhs_be.left,
                //                    &balanced,
                //                    lhs_be.op,
                //                    lhs_be.bool_modifier)?;

                let expr = BinaryOpExpr {
                    op: lhs_be.op,
                    left: lhs_be.left.clone(),
                    right: Box::new(balanced),
                    join_modifier: lhs_be.join_modifier.clone(),
                    group_modifier: lhs_be.group_modifier.clone(),
                    bool_modifier: lhs_be.bool_modifier,
                    span,
                };
                return Ok(
                    BinaryOperator(expr)
                );
            }
        }
        _ => {}
    }

    // validate_scalar_op(&lhs, &rhs, op, return_bool)?;

    let expr = BinaryOpExpr {
        op,
        left: Box::new(lhs),
        right: Box::new(rhs),
        join_modifier,
        group_modifier,
        bool_modifier: return_bool,
        span,
    };

    Ok(BinaryOperator(expr))
}


fn parse_group_modifier(p: &mut Parser) -> Result<GroupModifier, ParseError> {
    let tok = p.expect_one_of(&[TokenKind::Ignoring, TokenKind::On])?;

    let op: GroupModifierOp = match tok.kind {
        TokenKind::Ignoring => GroupModifierOp::Ignoring,
        TokenKind::On => GroupModifierOp::On,
        _ => unreachable!()
    };

    let args = p.parse_ident_list()?;
    Ok(GroupModifier::new(op, args))
}

fn parse_join_modifier(p: &mut Parser) -> ParseResult<JoinModifier> {
    let tok = p.expect_one_of(&[TokenKind::GroupLeft, TokenKind::GroupRight])?;

    let op = match tok.kind {
        TokenKind::GroupLeft => JoinModifierOp::GroupLeft,
        TokenKind::GroupRight => JoinModifierOp::GroupRight,
        _ => unreachable!()
    };

    let labels = if !p.at(TokenKind::LeftParen) {
        // join modifier may ignore ident list.
        vec![]
    } else {
        p.parse_ident_list()?
    };

    Ok( JoinModifier::new(op, labels) )
}

pub(super) fn parse_single_expr_without_rollup_suffix(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    match p.peek_kind() {
        StringLiteral => match parse_string_expr(p) {
            Ok(s) => Ok(Expression::String(s)),
            Err(e) => Err(e),
        },
        Ident => parse_ident_expr(p),
        Number | NumberWithUnit => parse_number_expr(p),
        Duration => parse_duration_expr(p),
        LeftBrace => parse_metric_expr(p),
        LeftParen => parse_parens_expr(p),
        OpPlus => parse_unary_plus_expr(p),
        OpMinus => parse_unary_minus_expr(p),
        _ => {
            Err(unexpected(p, "", "expression", None))
        }
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
    let val = duration.duration(1);
    if val < 0 {
        Err(ParseError::InvalidDuration(duration.text))
    } else {
        Ok(duration)
    }
}

fn parse_unary_plus_expr(p: &mut Parser) -> ParseResult<Expression> {
    p.expect(TokenKind::OpPlus)?;
    let expr = parse_single_expr(p)?;
    /**
    let t = checkAST(p, &expr)?;
    match t {
        ReturnType::Scalar | ReturnType::InstantVector => Ok(expr) {
        _ => {
            let msg = format!("unary expression only allowed on expressions of type scalar or instant vector, got {:?}", t);
            Err(ParseError::Gen
        }
    }
     **/
    Ok(expr)
}

fn parse_unary_minus_expr(p: &mut Parser) -> ParseResult<Expression> {
    // assert(p.at(TokenKind::Minus)
    let mut span = p.last_token_range().unwrap();
    p.bump();
    let expr = parse_single_expr(p)?;
    span = span.cover(expr.span());

    match expr.return_value() {
        ReturnValue::InstantVector |
        ReturnValue::Scalar => {}
        _ => {
            let msg = format!("unary expression only allowed on expressions of type scalar or instant vector");
            return Err(unexpected(p, "", &msg, Some(span)));
        }
    }

    /// Substitute `-expr` with `0 - expr`
    let lhs = Expression::Number(NumberExpr::new(0.0, span));
    let mut binop_expr = BinaryOpExpr::new(BinaryOp::Sub, lhs, expr)?;
    binop_expr.span = span;
    Ok(BinaryOperator(binop_expr))
}

pub(super) fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    use TokenKind::*;

    let mut ident_count = 0;
    let mut tokens: Vec<StringTokenType> = Vec::with_capacity(1);
    let mut span: TextSpan;
    let mut tok = p.expect_one_of(&[StringLiteral, Ident])?;

    loop {
        span = tok.span;

        match tok.kind {
            StringLiteral => {
                let str = extract_string_value(tok.text)?;
                tokens.push(StringTokenType::String(str));
                span = span.cover(tok.span)
            }
            Ident => {
                tokens.push(StringTokenType::Ident(tok.text.to_string()));
                span = span.cover(tok.span);
                ident_count += 1;
            }
            _ => {
                return Err(unexpected(p, "", "string literal", None));
            }
        }


        if p.at_end() {
            break;
        }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.

        tok = p.current_token()?;

        if tok.kind != OpPlus {
            break;
        }

        if let Some(token) = p.next_token() {
            tok = token;
        } else {
            // todo: err
            break
        }

        if tok.kind == StringLiteral {
            // "s1" + "s2"
            continue;
        }

        if tok.kind != Ident {
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

    // optimize if we have no idents
    if ident_count == 0 {
        let joined = tokens.iter().map(|x| {
            match x {
                StringTokenType::String(s) => s,
                _ => ""
            }
        }).collect::<Vec<_>>()
            .join("");

        return Ok(StringExpr::new(joined, span));
    }

    Ok(StringExpr::from_tokens(tokens, span))
}

pub(super) fn parse_parens_expr(p: &mut Parser) -> ParseResult<Expression> {
    let list = parse_arg_list(p)?;
    Ok(Expression::from(list))
}

pub(super) fn parse_arg_list(p: &mut Parser) -> ParseResult<Vec<BExpression>> {
    use TokenKind::*;
    p.expect(LeftParen)?;
    p.parse_comma_separated(&[RightParen], |p| {
        let expr = parse_expression(p)?;
        Ok(Box::new(expr))
    })
}

/// parses expressions starting with `ident` token.
fn parse_ident_expr(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    let tok = p.expect_token(Ident)?;

    // clone here to avoid issues with borrowing later
    // todo: how to avoid this, since its only needed in the function case
    let name = tok.text.clone();

    // Look into the next token in order to determine how to parse
    // the current expression.
    let kind = p.peek_kind();
    match kind {
        Eof | Offset => {
            p.back();
            return parse_metric_expr(p);
        }
        Ident => {
            p.back();
            if is_aggr_func(name) {
                return parse_aggr_func_expr(p);
            }
            return parse_metric_expr(p);
        }
        LeftParen => {
            p.back();
            return parse_function(p, name);
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
        }
    }

    Err(unexpected(p, "", "identifier", None))
}

fn is_aggr_func(name: &str) -> bool {
    AggregateFunction::from_str(name).is_ok()
}