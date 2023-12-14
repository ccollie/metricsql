use std::borrow::Cow;

use metricsql_parser::ast::Expr;
use metricsql_parser::label::{LabelFilter, Matchers};
use metricsql_parser::parser::parse;

use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(crate) fn join_matchers_with_extra_filters<'a>(
    src: &'a Matchers,
    etfs: &'a Vec<Vec<LabelFilter>>,
) -> Cow<'a, Matchers> {
    if src.is_empty() {
        return Cow::Owned::<'a>(Matchers::with_or_matchers(etfs.clone()));
    }
    if etfs.is_empty() {
        return Cow::Borrowed::<'a>(src);
    }

    let src_len = get_matcher_list_len(src);
    let mut filters: Vec<Vec<LabelFilter>> = Vec::with_capacity(src_len);
    for tf in src.iter() {
        for etf in etfs.iter() {
            let mut tfs = tf.clone();
            for filter in etf.iter() {
                tfs.push(filter.clone())
            }
            filters.push(tfs)
        }
    }

    let dst = Matchers::with_or_matchers(filters);
    Cow::Owned::<'a>(dst)
}

fn get_matcher_list_len(matchers: &Matchers) -> usize {
    let mut len = 0;
    if !matchers.matchers.is_empty() {
        len += 1;
    }
    len += matchers.or_matchers.len();
    len
}


/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RuntimeResult<Matchers> {
    match parse(s) {
        Ok(expr) => match expr {
            Expr::MetricExpression(me) => {
                if me.is_empty() {
                    let msg = "labelFilters cannot be empty";
                    return Err(RuntimeError::from(msg));
                }
                Ok(me.matchers)
            }
            _ => {
                let msg = format!("expecting metric selector; got {}", expr);
                Err(RuntimeError::from(msg))
            }
        },
        Err(err) => Err(RuntimeError::ParseError(err)),
    }
}
