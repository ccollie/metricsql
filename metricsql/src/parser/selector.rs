use crate::ast::{Expr, MetricExpr, WithArgExpr};
use crate::common::{LabelFilter, LabelFilterExpr, LabelFilterOp, NAME_LABEL};
use crate::parser::expr::parse_string_expr;
use crate::parser::{ParseError, ParseResult, Parser};
use super::tokens::Token;
use std::collections::HashSet;
use crate::parser::parse_error::unexpected;

/// parse_metric_expr parses a metric.
///
///		<label_set>
///		<metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expr> {
    let mut me = MetricExpr::default();

    if p.at(&Token::Identifier) {
        let token = p.expect_identifier()?;
        let resolved = resolve_metric_name(p, &token)?.unwrap_or(token);

        let filter =
            LabelFilter::new(LabelFilterOp::Equal, NAME_LABEL, resolved)?;
        me.label_filters.push(filter);

        if !p.at(&Token::LeftBrace) {
            return Ok(Expr::MetricExpression(me));
        }
    }
    let filters = parse_label_filters(p)?;
    for filter in filters.into_iter() {
        if !filter.is_resolved() {
            me.label_filter_expressions.push(filter);
        } else {
            me.label_filters.push(filter.to_label_filter()?);
        }
    }

    Ok(Expr::MetricExpression(me))
}

/// parse_label_filters parses a set of label matchers.
///
///		'{' [ <label_name> <match_op> <match_string>, ... ] '}'
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<LabelFilterExpr>> {
    use Token::*;
    p.expect(&LeftBrace)?;
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
    use Token::*;

    let label = p.expect_identifier()?;
    let tok = p.current_token()?;

    if tok.kind == RightBrace {
        // we have something like
        // WITH (x = {foo="bar"}) metric{x}
        // label here would be 'x'
        return if let Some(wae) = p.lookup_with_expr(&label) {
            match &wae.expr {
                Expr::MetricExpression(me) => {
                    if has_non_empty_metric_group(&me) {
                        // we have something like WITH (x = cpu{foo="bar"}) metric{x}
                        // which we don't allow
                        // return Err()
                        let msg = format!(
                            "cannot expand a selector with a metric group ({:?}) to a label filter",
                            me
                        );
                        return Err(ParseError::General(msg));
                    }

                    Ok(me.label_filter_expressions.clone())
                }
                _ => Err(unexpected("label filter",
                                    &wae.expr.to_string(),
                                    "selector Expr",
                                    Some(&tok.span)))
            }
        } else {
            let err = format!("variable {} not found in Expr", label);
            Err(ParseError::General(err))
        };
    }

    let op = match tok.kind {
        Equal => LabelFilterOp::Equal,
        OpNotEqual => LabelFilterOp::NotEqual,
        RegexEqual => LabelFilterOp::RegexEqual,
        RegexNotEqual => LabelFilterOp::RegexNotEqual,
        _ => return Err(
            unexpected(
                "label filter",
                &tok.kind.to_string(),
                "=, !=, =~ or !~",
                Some(&tok.span)
            )
        ),
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    let value = parse_string_expr(p)?;
    let filter = LabelFilterExpr {
        op,
        label,
        value
    };
    // todo: validate if regex
    Ok(vec![filter])
}

fn dedupe_label_filters(lfs: &mut Vec<LabelFilterExpr>) {
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
fn resolve_metric_name(p: &Parser, name: &str) -> ParseResult<Option<String>> {
    let wa = p.lookup_with_expr(name);
    if wa.is_none() {
        return Ok(None);
    }

    let wa = wa.unwrap();
    // todo: handle wa.args.len() > 0

    // let e_new = expand_with_expr_ext(was, wa, None)?;

    let handle_metric_expr = |me: &MetricExpr, wa: &WithArgExpr| -> ParseResult<String> {
        if !is_only_metric_group(me) {
            let msg = format!(
                "cannot expand {:?} to non-metric Expr {:?}",
                me, wa.expr
            );
            return Err(ParseError::General(msg));
        }
        Ok(me.name().unwrap_or_default().to_string())
    };

    match &wa.expr {
        Expr::StringExpr(se) => {
            return Ok(Some(se.to_string()));
        }
        Expr::StringLiteral(_se) => {
            // let resolved = resolve()
            // return Ok(se.to_string())
        }
        Expr::MetricExpression(me) => {
            return Ok(Some(handle_metric_expr(me, wa)?));
        }
        _ => {}
    };

    let msg = format!(
        "cannot resolve {} as string parsing metric name. Found {:?}",
        name, wa.expr
    );
    Err(ParseError::General(msg))
}


pub fn is_only_metric_group(me: &MetricExpr) -> bool {
    if !has_non_empty_metric_group(me) {
        return false;
    }
    if !me.label_filters.is_empty() {
        me.label_filters.len() == 1
    } else {
        me.label_filter_expressions.len() == 1
    }
}

fn has_non_empty_metric_group(me: &MetricExpr) -> bool {
    if !me.label_filters.is_empty() {
        // we should have at least the name filter
        return me.label_filters[0].is_metric_name_filter();
    }
    if me.label_filter_expressions.is_empty() {
        return false;
    }
    me.label_filter_expressions[0].is_metric_name_filter()
}