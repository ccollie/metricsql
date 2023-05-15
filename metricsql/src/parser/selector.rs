use super::tokens::Token;
use crate::ast::{Expr, MetricExpr};
use crate::common::{LabelFilterExpr, LabelFilterOp, StringExpr, NAME_LABEL};
use crate::parser::expr::parse_string_expr;
use crate::parser::parse_error::unexpected;
use crate::parser::{ParseResult, Parser};
use std::collections::HashSet;
use crate::prelude::LabelFilter;

/// parse_metric_expr parses a metric.
///
///		<label_set>
///		<metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expr> {
    let mut me = MetricExpr::default();
    let can_expand = p.can_lookup();

    if p.at(&Token::Identifier) {
        let token = p.expect_identifier()?;

        if can_expand {
            p.needs_expansion = true;
            let filter =
                LabelFilterExpr::new(LabelFilterOp::Equal, NAME_LABEL, StringExpr::from(token))?;

            me.label_filter_expressions.push(filter);
        } else {
            let filter = LabelFilter::new(LabelFilterOp::Equal, NAME_LABEL, token)?;
            me.label_filters.push(filter);
        }

        if !p.at(&Token::LeftBrace) {
            return Ok(Expr::MetricExpression(me));
        }
    }

    let filters = parse_label_filters(p)?;
    // symbol table is empty and we're not parsing a WITH statement
    if !can_expand {
        for filter in filters {
            let resolved = filter.to_label_filter()?;
            me.label_filters.push(resolved);
        }
    } else {
        p.needs_expansion = true;
        me.label_filter_expressions.extend_from_slice(&filters);
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
    let mut filters = p.parse_comma_separated(&[RightBrace], parse_label_filter)?;
    dedupe_label_filters(&mut filters);
    Ok(filters)
}

/// parse_label_filter parses a single label matcher.
///
///   <label_name> <match_op> <match_string> | identifier
///
fn parse_label_filter(p: &mut Parser) -> ParseResult<LabelFilterExpr> {
    use Token::*;

    let mut filter: LabelFilterExpr = LabelFilterExpr::default();
    filter.label = p.expect_identifier()?;
    filter.op = LabelFilterOp::Equal;

    let tok = p.current_token()?;
    match tok.kind {
        Equal => filter.op = LabelFilterOp::Equal,
        OpNotEqual => filter.op = LabelFilterOp::NotEqual,
        RegexEqual => filter.op = LabelFilterOp::RegexEqual,
        RegexNotEqual => filter.op = LabelFilterOp::RegexNotEqual,
        Comma | RightBrace => return Ok(filter),
        _ => {
            return Err(unexpected(
                "label filter",
                &tok.text,
                "=, !=, =~ or !~",
                Some(&tok.span),
            ))
        }
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    filter.value = parse_string_expr(p)?;

    // todo: validate if regex
    Ok(filter)
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
