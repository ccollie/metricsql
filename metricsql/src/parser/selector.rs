use enquote::{unescape};
use crate::ast::{Expression, LabelFilter, LabelFilterExpr, LabelFilterOp, MetricExpr, NAME_LABEL};
use crate::lexer::{TokenKind, unescape_ident};
use crate::parser::{ParseError, Parser, ParseResult};
use crate::parser::expr::parse_string_expr;
use crate::parser::parser::unexpected;

/// parse_metric_expr parses a metric.
///
///		<label_set>
///		<metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expression> {
    let mut me= MetricExpr::default();

    let mut span = p.last_token_range().unwrap();
    if p.at(TokenKind::Ident) {
        let tok = p.current_token()?;

        let token = match unescape(tok.text, None) {
            Err(_) => {
                return Err(ParseError::General(
                    format!("Invalid selector name : {}", tok.text)
                ));
            },
            Ok(value) => value
        };

        let filter = LabelFilter::new(
            LabelFilterOp::Equal, NAME_LABEL, &token)?;

        me.label_filters.push(filter);

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

/// parse_label_filters parses a set of label matchers.
///
///		'{' [ <label_name> <match_op> <match_string>, ... ] '}'
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;
    p.expect(LeftBrace)?;
    let vec = p.parse_comma_separated(&[RightBrace], parse_label_filter_expr)?;
    let iter = vec.into_iter().flatten().collect::<Vec<LabelFilterExpr>>();
    Ok(iter)
}

fn parse_label_filter_expr(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;

    let token = p.expect_token(Ident)?;
    let label = unescape_ident(token.text);
    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{ident}
    let tok = p.current_token()?;

    if tok.kind == RightBrace {
        if !p.is_parsing_with() {
            return Err(unexpected(p, "label filter", "=, !=, =~ or !~", None))
        }
        // we have something like
        // WITH (x = {foo="bar"}) metric{x}
        // label here would be 'x'
        return if let Some(wae) = p.lookup_with_expr(&label) {
            // return
            match &*wae.expr {
                Expression::MetricExpression(me) => {
                    if me.has_non_empty_metric_group() {
                        // we have something like WITH (x = cpu{foo="bar"}) metric{x}
                        // which we don't allow
                        // return Err()
                    }

                    Ok(me.label_filter_exprs.clone())
                }
                _ => {
                    Err(unexpected(p, "label filter", "selector expression", None))
                }
            }
        } else {
            let err = format!("variable {} not found in WITH expression", label);
            Err(ParseError::General(err))
        }
    }

    let op = match tok.kind {
        Equal => LabelFilterOp::Equal,
        OpNotEqual => LabelFilterOp::NotEqual,
        RegexEqual => LabelFilterOp::RegexEqual,
        RegexNotEqual => LabelFilterOp::RegexNotEqual,
        _ => {
            return Err(unexpected(p, "label filter", "=, !=, =~ or !~", None))
        }
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    let se = parse_string_expr(p)?;
    Ok(vec![LabelFilterExpr::new(label, se, op)])
}