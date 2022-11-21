use std::str::FromStr;
use crate::ast::{BExpression, BinaryOp, BinaryOpExpr, DurationExpr, Expression, GroupModifier, GroupModifierOp, JoinModifier, JoinModifierOp, NumberExpr, ReturnValue, StringExpr, StringTokenType};
use crate::functions::AggregateFunction;
use crate::lexer::{parse_float, TokenKind};
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::parser::unexpected;
use super::aggregation::parse_aggr_func_expr;
use super::function::parse_function;
use super::rollup::parse_rollup_expr;
use super::selector::parse_metric_expr;
use super::with_expr::parse_with_expr;


pub(super) fn parse_number(p: &mut Parser) -> ParseResult<f64> {
    let tok = p.expect_one_of(&[TokenKind::Number, TokenKind::NaN, TokenKind::Inf])?;
    parse_float(tok.text)
}

pub(super) fn parse_number_expr(p: &mut Parser) -> ParseResult<Expression> {
    let tok = p.expect_one_of(&[TokenKind::Number, TokenKind::NaN, TokenKind::Inf])?;
    let value = parse_float(tok.text)?;
    Ok(Expression::Number(NumberExpr::new(value, tok.span)))
}

pub(super) fn parse_duration(p: &mut Parser) -> ParseResult<DurationExpr> {
    let tok = p.expect_one_of(&[TokenKind::Number, TokenKind::Duration])?;
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

    let start = left.span();

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
        let span = start.cover(right.span());

        let mut be = BinaryOpExpr::new(operator, left, right)?;
        be.group_modifier = group_modifier;
        be.join_modifier = join_modifier;
        be.bool_modifier = is_bool;
        be.span = span;

        // try to avoid an unnecessary clone
        if should_balance_binary_op(&be) {
            left = balance_binary_op(&be, false);
        } else {
            left = Expression::BinaryOperator(be)
        }
    }

    Ok(left)
}


fn should_balance_binary_op(be: &BinaryOpExpr) -> bool {
    match &be.left.as_ref() {
        Expression::BinaryOperator(left) => {
            let lp = left.op.precedence();
            let rp = be.op.precedence();
            if rp < lp || (rp == lp && !be.op.is_right_associative()) {
                false
            } else {
                true
            }
        },
        _ => false
    }
}

fn balance_binary_op(be: &BinaryOpExpr, first: bool) -> Expression {
    if first || should_balance_binary_op(&be) {
        let mut res = be.clone();
        let temp = res.left;
        res.left = res.right;
        res.right = temp;

        // std::mem::swap(&mut res.left, &mut res.right);
        let balanced = balance_binary_op(&res, false);
        res.right = Box::new(balanced);
    }
    Expression::BinaryOperator(be.clone())
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

    let mut res = JoinModifier::new(op);
    if !p.at(TokenKind::LeftParen) {
        // join modifier may ignore ident list.
    } else {
        res.labels = p.parse_ident_list()?;
    }

    Ok(res)
}

pub(super) fn parse_single_expr_without_rollup_suffix(p: &mut Parser) -> ParseResult<Expression> {
    use TokenKind::*;

    match p.peek_kind() {
        LiteralString => match parse_string_expr(p) {
            Ok(s) => Ok(Expression::String(s)),
            Err(e) => Err(e),
        },
        Ident => parse_ident_expr(p),
        Inf | NaN | Number => parse_number_expr(p),
        Duration => parse_duration_expr(p),
        LeftBrace => parse_metric_expr(p),
        LeftParen => parse_parens_expr(p),
        OpPlus => parse_unary_plus_expr(p),
        OpMinus => parse_unary_minus_expr(p),
        _ => {
            Err( unexpected(p, "", "expression", None) )
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
    let e = parse_single_expr(p)?;
    span = span.cover(e.span());
    match e.return_value() {
        ReturnValue::InstantVector |
        ReturnValue::Scalar => {
            // handle special cases
            match e {
                Expression::Number(n) => {
                    let v = n.value;
                    if v.is_finite() {
                        return Ok(Expression::from(-v));
                    } else if v == f64::INFINITY {
                        return Ok(Expression::from(f64::NEG_INFINITY));
                    }
                },
                _ => {}
            }
        },
        _ => {
            let msg = format!("unary expression only allowed on expressions of type scalar or instant vector");
            return Err(unexpected(p,"", &msg, Some(span)));
        }
    }
    // Substitute `-expr` with `0 - expr`
    let b = BinaryOpExpr::new_unary_minus(e, span)?;
    Ok(Expression::BinaryOperator(b))
}

pub(super) fn parse_string_expr(p: &mut Parser) -> ParseResult<StringExpr> {
    use TokenKind::*;

    let mut tokens: Vec<StringTokenType> = Vec::with_capacity(1);
    let mut tok = p.expect_token(LiteralString)?;
    let mut span = tok.span;
    let mut ident_count = 0;

    loop {
        match tok.kind {
            LiteralString => {
                if tok.text.len() == 2 {
                    tokens.push(StringTokenType::String("".to_string()));
                } else {
                    let slice = &tok.text[1 .. tok.text.len() - 1];
                    tokens.push(StringTokenType::String(slice.to_string()));
                }
                span = span.cover(tok.span)
            }
            Ident => {
                tokens.push(StringTokenType::Ident(tok.text.to_string()));
                span = span.cover(tok.span);
                ident_count += 1;
            }
            _ => {
                return Err(unexpected(p, "", "string literal", None))
            }
        }

        // composite StringExpr like `"s1" + "s2"`, `"s" + m()` or `"s" + m{}` or `"s" + unknownToken`.

        tok = p.current_token()?;

        if tok.kind != OpPlus {
            break;
        }

        if tok.kind == LiteralString {
            // "s1" + "s2"
            continue;
        }

        if tok.kind != Ident {
            // "s" + unknownToken
            p.back();
            break
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

    // optimize if w have no idents
    if ident_count == 0 {
        let joined = tokens.iter().map(|x| {
            match x {
               StringTokenType::String(s) => s,
                _=> ""
            }
        }).collect::<Vec<_>>()
            .join("");

        return Ok( StringExpr::new(joined, span) )
    }

    Ok( StringExpr::from_tokens(tokens, span) )
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
    let name = tok.text.clone();

    // Look into the next token in order to determine how to parse
    // the current expression.
    let tok = p.current_token()?;
    let kind = tok.kind;
    match kind {
        Eof | Offset => {
            p.back();
            return parse_metric_expr(p);
        }
        Ident => {
            p.back();
            if is_aggr_func(name) {
                return parse_aggr_func_expr(p)
            }
            return parse_metric_expr(p);
        }
        LeftParen => {
            p.back();
            // unwrap is safe here since LeftParen is the previous token
            // clone is to stop holding on to p
            return parse_function(p, name)
        }
        LeftBrace | LeftBracket | RightParen | Comma | At => {
            p.back();
            return parse_metric_expr(p)
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