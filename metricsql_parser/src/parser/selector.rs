use crate::ast::{Expr, InterpolatedSelector, MetricExpr};
use crate::label::{LabelFilterExpr, LabelFilterOp};
use crate::parser::{Parser, ParseResult};
use crate::parser::expr::parse_string_expr;
use crate::parser::parse_error::unexpected;

use super::tokens::Token;

/// parse_metric_expr parses a metric.
///
///    <label_set>
///    <metric_identifier> [<label_set>]
///
pub fn parse_metric_expr(p: &mut Parser) -> ParseResult<Expr> {
    let can_expand = p.can_lookup();
    let mut name: Option<String> = None;

    fn create_metric_expr(
        name: Option<String>,
        filters: Vec<Vec<LabelFilterExpr>>,
    ) -> ParseResult<Expr> {

        let mut me = if let Some(name) = name {
            MetricExpr::new(name)
        } else {
            MetricExpr::default()
        };

        if filters.is_empty() {
            return Ok(Expr::MetricExpression(me));
        }

        if filters.len() == 1 {
            if let Some(first) = filters.first() {
                if first.is_empty() {
                    return Ok(Expr::MetricExpression(me));
                }
                let converted = first.iter().map(|x| x.to_label_filter()).collect::<ParseResult<Vec<_>>>()?;
                me.matchers.matchers = converted;
                me.sort_filters();
            }
            return Ok(Expr::MetricExpression(me));
        }

        let mut or_matchers = vec![];
        for filter in filters {
            let converted = filter.iter()
                .map(|x| x.to_label_filter())
                .collect::<ParseResult<Vec<_>>>()?;
            or_matchers.push(converted);
        }

        me.matchers.or_matchers = or_matchers;
        me.sort_filters();
        Ok(Expr::MetricExpression(me))
    }

    if p.at(&Token::Identifier) {
        let token = p.expect_identifier()?;

        if !p.at(&Token::LeftBrace) {
            let me = MetricExpr::new(token);
            return Ok(Expr::MetricExpression(me));
        }

        name = Some(token.to_string());
    }

    let filters = parse_label_filters(p)?;
    // symbol table is empty and we're not parsing a WITH statement
    if !can_expand {
        create_metric_expr(name, filters)
    } else {
        // no identifiers in the label filters, create a MetricExpr
        if filters.iter().all(|x| x.iter().all(|filter| filter.is_resolved())) {
            return create_metric_expr(name, filters);
        }
        p.needs_expansion = true;
        let mut with_me = if let Some(name) = name {
            InterpolatedSelector::new(name)
        } else {
            InterpolatedSelector::default()
        };
        with_me.matchers = filters;
        Ok(Expr::WithSelector(with_me))
    }
}

/// parse_label_filters parses a set of label matchers.
///
/// '{' [ <label_name> <match_op> <match_string>, ... ] '}'
///
fn parse_label_filters(p: &mut Parser) -> ParseResult<Vec<Vec<LabelFilterExpr>>> {
    use Token::*;

    p.expect(&LeftBrace)?;
    let mut result: Vec<Vec<LabelFilterExpr>> = Vec::with_capacity(2);

    loop {
        let filters = p.parse_comma_separated(&[RightBrace, OpOr], parse_label_filter)?;
        if !p.can_lookup() {
            // if we're not parsing a WITH statement, we need to make sure we have no unresolved identifiers
            for filter in &filters {
                if filter.is_variable() {
                    return Err(unexpected(
                        "label filter",
                        &filter.label,
                        "unresolved identifier",
                        None,
                    ));
                }
            }
        }

        if !filters.is_empty() {
            result.push(filters);
        }

        if !p.at(&OpOr) {
            break;
        }
    }

    Ok(result)
}

/// parse_label_filter parses a single label matcher.
///
///   <label_name> <match_op> <match_string> | identifier
///
fn parse_label_filter(p: &mut Parser) -> ParseResult<LabelFilterExpr> {
    use Token::*;

    let label = p.expect_identifier()?;
    let op: LabelFilterOp;

    let tok = p.current_token()?;
    match tok.kind {
        Equal => op = LabelFilterOp::Equal,
        OpNotEqual => op = LabelFilterOp::NotEqual,
        RegexEqual => op = LabelFilterOp::RegexEqual,
        RegexNotEqual => op = LabelFilterOp::RegexNotEqual,
        Comma | RightBrace => return Ok(LabelFilterExpr::variable(&label)),
        _ => {
            return Err(unexpected(
                "label filter",
                tok.text,
                "=, !=, =~ or !~",
                Some(&tok.span),
            ))
        }
    };

    p.bump();

    // todo: if we're parsing a WITH, we can accept an ident. IOW, we can have metric{s=ident}
    let value = parse_string_expr(p)?;

    LabelFilterExpr::new(label, op, value)
}
