use std::collections::{HashMap, HashSet, Vec};
use crate::error::{Error, Result};
use crate::lexer::{Lexer, quote, Token, TokenKind, unescape_ident};
use crate::parser::expand_with::expand_with_expr;
use crate::types::*;

fn parse_metric_expr(mut lex: &Lexer) -> Result<MetricExpr> {
    let mut me = MetricExpr {
        label_filters: vec![],
        label_filter_exprs: vec![],
        span: lex.span()
    };
    let mut tok = lex.token().unwrap();
    if tok.kind == TokenKind::Ident {
        let tokens = vec![quote(&unescape_ident(tok.text))];
        let value = StringExpr { s: "".as_str(), tokens: Some(tokens) };
        let lfe = LabelFilterExpr { label: "__name__".as_str(), value, op: types::LabelFilterOp::Equal };
        me.label_filter_exprs.push(lfe);

        tok = lex.next().unwrap();
        if tok.kind != TokenKind::LeftBrace {
            return Ok(me);
        }
    }
    me.label_filter_exprs = parse_label_filters(lex)?;
    Ok(me)
}