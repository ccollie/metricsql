use crate::ast::{DurationExpr, Expression, ReturnValue, RollupExpr};
use crate::lexer::TokenKind;
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::expr::parse_single_expr_without_rollup_suffix;
use super::expr::{parse_duration, parse_positive_duration};


pub(super) fn parse_rollup_expr(p: &mut Parser, e: Expression) -> ParseResult<Expression> {
    let mut re = RollupExpr::new(e);

    let mut at: Option<Expression> = None;
    if p.at(TokenKind::LeftBracket) {
        let (window, step, inherit_step) = parse_window_and_step(p)?;
        re.window = window;
        re.step = step;
        re.inherit_step = inherit_step;
    }

    if p.at(TokenKind::At) {
        at = Some(parse_at_expr(p)?);
    }

    if p.at(TokenKind::Offset) {
        re.offset = Some(parse_offset(p)?);
    }

    if p.at(TokenKind::At) {
        if at.is_some() {
            let msg = "RollupExpr: duplicate '@' token".to_string();
            return Err(ParseError::General(msg));
        }
        at = Some(parse_at_expr(p)?);
    }

    if let Some(v) = at {
        re.at = Some(Box::new(v))
    }

    p.update_span(&mut re.span);

    Ok(Expression::Rollup(re))
}

fn parse_at_expr(p: &mut Parser) -> ParseResult<Expression> {
    p.expect(TokenKind::At)?;
    let expr = parse_single_expr_without_rollup_suffix(p)?;
    // validate result type
    match expr.return_value() {
        ReturnValue::InstantVector | ReturnValue::Scalar => Ok(expr),
        ReturnValue::Unknown(cause) => {
            // todo: pass span
            Err(ParseError::InvalidExpression(cause.message))
        }
        _ => Err(ParseError::InvalidExpression(
            String::from("@ modifier expression must return a scalar or instant vector")
        )) // todo: have InvalidReturnType enum variant
    }
}

fn parse_window_and_step(
    p: &mut Parser,
) -> Result<(Option<DurationExpr>, Option<DurationExpr>, bool), ParseError> {
    p.expect(TokenKind::LeftBracket)?;

    let mut window: Option<DurationExpr> = None;

    if !p.at(TokenKind::Colon) {
        window = Some(parse_positive_duration(p)?);
    }

    let mut step: Option<DurationExpr> = None;
    let mut inherit_step = false;

    if p.at(TokenKind::Colon) {
        p.bump();
        // Parse step
        if p.at(TokenKind::RightBracket) {
            inherit_step = true;
        }
        if !p.at(TokenKind::RightBracket) {
            step = Some(parse_positive_duration(p)?);
        }
    }
    p.expect(TokenKind::RightBracket)?;

    Ok((window, step, inherit_step))
}

fn parse_offset(p: &mut Parser) -> ParseResult<DurationExpr> {
    p.expect(TokenKind::Offset)?;
    parse_duration(p)
}