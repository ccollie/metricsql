use crate::ast::{Expression, LabelFilterExpr, LabelFilterOp, MetricExpr, NAME_LABEL};
use crate::lexer::{TokenKind, unescape_ident};
use crate::parser::{Parser, ParseResult};
use crate::parser::expr::parse_string_expr;

pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expression> {
    let mut me= MetricExpr::default();

    let mut span = p.last_token_range().unwrap();
    if p.at(TokenKind::Ident) {
        let tok = p.current_token()?;

        let token = unescape_ident(tok.text);
        let lfe = LabelFilterExpr::new_tag(NAME_LABEL, LabelFilterOp::Equal, &token, span);
        me.label_filter_exprs.push(lfe);

        p.bump();
        if !p.at(TokenKind::LeftBrace) {
            return Ok(Expression::MetricExpression(me));
        }
    }
    let lfes = parse_label_filters(p)?;
    me.label_filter_exprs.extend(lfes.into_iter());
    p.update_span(&mut span);
    me.span = span;
    Ok(Expression::MetricExpression(me))
}

fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;
    p.expect(LeftBrace)?;
    p.parse_comma_separated(&[RightBrace], parse_label_filter_expr)
}

fn parse_label_filter_expr(p: &mut Parser) -> ParseResult<LabelFilterExpr> {
    use TokenKind::*;

    let token = p.expect_token(Ident)?;
    let label = unescape_ident(token.text);

    let op_token = p.expect_one_of(&[Equal, OpNotEqual, RegexEqual, RegexNotEqual])?;

    let op = match op_token.kind {
        Equal => LabelFilterOp::Equal,
        OpNotEqual => LabelFilterOp::NotEqual,
        RegexEqual => LabelFilterOp::RegexEqual,
        RegexNotEqual => LabelFilterOp::RegexNotEqual,
        _ => unreachable!()
    };

    let se = parse_string_expr(p)?;
    Ok(LabelFilterExpr::new(label, se, op))
}