use crate::ast::label_filter_expr::LabelFilterExpr;
use crate::ast::{Expression, MetricExpr, StringExpr, WithArgExpr};
use crate::common::{LabelFilterOp, NAME_LABEL};
use crate::lexer::{unescape_ident, TokenKind};
use crate::parser::expr::parse_string_expr;
use crate::parser::parser::unexpected;
use crate::parser::{ParseError, ParseResult, Parser};
use enquote::unescape;
use std::collections::HashSet;

/// parse_metric_expr parses a metric.
///
///		<label_set>
///		<metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expression> {
    let mut me = MetricExpr::default();

    if p.at(TokenKind::Ident) {
        let tok = p.current_token()?;

        let token = match unescape(tok.text, None) {
            Err(_) => {
                return Err(ParseError::General(format!(
                    "Invalid selector name : {}",
                    tok.text
                )));
            }
            Ok(value) => value,
        };

        let name = resolve_metric_name(p, &token)?;
        let filter =
            LabelFilterExpr::new(LabelFilterOp::Equal, NAME_LABEL, StringExpr::from(name))?;

        me.label_filters.push(filter);

        p.bump();
        if !p.at(TokenKind::LeftBrace) {
            return Ok(Expression::MetricExpression(me));
        }
    }
    let filters = parse_label_filters(p)?;
    me.label_filters.extend(filters.into_iter());
    Ok(Expression::MetricExpression(me))
}

/// parse_label_filters parses a set of label matchers.
///
///		'{' [ <label_name> <match_op> <match_string>, ... ] '}'
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;
    p.expect(LeftBrace)?;
    let vec = p.parse_comma_separated(&[RightBrace], parse_label_filter)?;
    let mut filters = vec.into_iter().flatten().collect::<Vec<LabelFilterExpr>>();

    dedupe_label_filters(&mut filters);
    Ok(filters)
}

/// parse_label_filter parses a single label matcher.
///
///   <label_name> <match_op> <match_string> | identifier
///
fn parse_label_filter(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use TokenKind::*;

    let token = p.expect_token(Ident)?;
    let label = unescape_ident(token.text);

    let tok = p.current_token()?;

    if tok.kind == RightBrace {
        // we have something like
        // WITH (x = {foo="bar"}) metric{x}
        // label here would be 'x'
        return if let Some(wae) = p.lookup_with_expr(&label) {
            match &wae.expr {
                Expression::MetricExpression(me) => {
                    if me.has_non_empty_metric_group() {
                        // we have something like WITH (x = cpu{foo="bar"}) metric{x}
                        // which we don't allow
                        // return Err()
                        let msg = format!(
                            "cannot expand a selector with a metric group ({:?}) to a label filter",
                            me
                        );
                        return Err(ParseError::General(msg));
                    }

                    Ok(me.label_filters.clone())
                }
                _ => Err(unexpected(p, "label filter", "selector expression", None)),
            }
        } else {
            let err = format!("variable {} not found in expression", label);
            Err(ParseError::General(err))
        };
    }

    let op = match tok.kind {
        Equal => LabelFilterOp::Equal,
        OpNotEqual => LabelFilterOp::NotEqual,
        RegexEqual => LabelFilterOp::RegexEqual,
        RegexNotEqual => LabelFilterOp::RegexNotEqual,
        _ => return Err(unexpected(p, "label filter", "=, !=, =~ or !~", None)),
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    let se = parse_string_expr(p)?;
    let filter = LabelFilterExpr {
        op,
        label,
        value: se,
    };
    // todo: validate if regex
    Ok(vec![filter])
}

pub fn dedupe_label_filters(lfs: &mut Vec<LabelFilterExpr>) {
    let mut set: HashSet<String> = HashSet::with_capacity(lfs.len());
    lfs.retain(|lf| {
        let key = lf.to_string();
        if set.contains(&key) {
            return false;
        }
        set.insert(key);
        true
    })
}

// todo: COW
fn resolve_metric_name(p: &Parser, name: &str) -> ParseResult<String> {
    let wa = p.lookup_with_expr(name);
    if wa.is_none() {
        return Ok(name.to_string());
    }

    let wa = wa.unwrap();
    // todo: handle wa.args.len() > 0

    // let e_new = expand_with_expr_ext(was, wa, None)?;

    let handle_metric_expr = |me: &MetricExpr, wa: &WithArgExpr| -> ParseResult<String> {
        if !me.is_only_metric_group() {
            let msg = format!(
                "cannot expand {:?} to non-metric expression {:?}",
                me, wa.expr
            );
            return Err(ParseError::General(msg));
        }
        Ok(me.name().unwrap_or_default())
    };

    match &wa.expr {
        Expression::String(_se) => {
            // let resolved = resolve()
            // return Ok(se.to_string())
        }
        Expression::MetricExpression(me) => {
            return Ok(handle_metric_expr(me, wa)?);
        }
        _ => {}
    };

    let msg = format!(
        "cannot resolve {} as string parsing metric name. Found {:?}",
        name, wa.expr
    );
    Err(ParseError::General(msg))
}
