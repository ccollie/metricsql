use std::borrow::Cow;

use metricsql::ast::Expr;
use metricsql::common::{LabelFilter, Matchers};
use metricsql::parser::parse;

use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(crate) fn join_matchers_vec<'a>(
    src: &'a Vec<Matchers>,
    etfs: &'a Vec<Matchers>,
) -> Cow<'a, Vec<Matchers>> {
    if src.is_empty() {
        return Cow::Borrowed::<'a>(etfs);
    }
    if etfs.is_empty() {
        return Cow::Borrowed::<'a>(src);
    }
    let mut dst: Vec<Matchers> = Vec::with_capacity(src.len());
    for tf in src.iter() {
        dst.push(tf.clone());
    }
    Cow::Owned::<'a>(dst)
}

/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RuntimeResult<Vec<LabelFilter>> {
    match parse(s) {
        Ok(expr) => match expr {
            Expr::MetricExpression(me) => {
                if me.is_empty() {
                    let msg = "labelFilters cannot be empty";
                    return Err(RuntimeError::from(msg));
                }
                Ok(me.label_filters)
            }
            _ => {
                let msg = format!("expecting metric selector; got {}", expr);
                Err(RuntimeError::from(msg))
            }
        },
        Err(err) => Err(RuntimeError::ParseError(err)),
    }
}
