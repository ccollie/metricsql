use std::collections::HashSet;
use std::ops::Deref;
use std::sync::Arc;
use std::string::String;

use chrono::Utc;

use lib::{round_to_decimal_digits};
use metricsql::ast::Expression;
use metricsql::parser::visit_all;

use crate::context::Context;
use crate::eval::{EvalConfig, Evaluator};
use crate::functions::types::AnyValue;
use crate::parser_cache::ParseCacheValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::QueryResult;
use crate::timeseries::Timeseries;

pub fn parse_promql_with_cache(ctx: &mut Context, q: &str) -> RuntimeResult<Arc<ParseCacheValue>> {
    ctx.parse_promql(q)
}

pub(crate) fn exec_internal(
    context: &Context,
    ec: &mut EvalConfig,
    q: &str) -> RuntimeResult<(AnyValue, Arc<ParseCacheValue>)> {

    let start_time = Utc::now();
    if context.config.stats_enabled && context.query_stats.is_enabled() {
        defer! {
            context.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = context.parse_promql(q)?;
    match (&parsed.evaluator, &parsed.has_subquery) {
        (Some(evaluator), has_subquery) => {

            if *has_subquery {
                ec.ensure_timestamps()?;
            }

            let ctx = Arc::new(context);
            let qid = context.active_queries.register_with_start(ec, q, start_time.timestamp());
            let rv = evaluator.eval(&ctx, ec);
            match rv {
                Ok(rv) => {
                    context.active_queries.remove(qid);
                    Ok((rv, Arc::clone(&parsed)))
                },
                Err(err) => {
                    context.active_queries.remove(qid);
                    return Err(err);
                }
            }
        },
        _ => {
            panic!("Bug: Invalid parse state")
        }
    }
}

/// executes q for the given config.
pub fn exec(context: &Context,
            ec: &mut EvalConfig,
            q: &str,
            is_first_point_only: bool) -> RuntimeResult<Vec<QueryResult>> {

    let (rv, parsed) = exec_internal(context, ec, q)?;

    let mut rv = rv.into_instant_vector(ec)?;
    if is_first_point_only {
        if rv[0].timestamps.len() > 0 {
            let timestamps = Arc::new(vec![rv[0].timestamps[0]]);
            // Remove all the points except the first one from every time series.
            for ts in rv.iter_mut() {
                ts.values.resize(1, f64::NAN);
                ts.timestamps = Arc::clone(&timestamps);
            }
        }
    }

    // at this point, parsed.expr is Some, but lets avoid unwrap in any case
    let may_sort = if let Some(expr) = &parsed.expr {
        may_sort_results(&expr, &rv)
    } else {
        false
    };

    let mut result = timeseries_to_result(&mut rv, may_sort)?;

    let n = ec.round_digits as u8;
    if n < 100 {
        for r in result.iter_mut() {
            for v in r.values.iter_mut() {
                *v = round_to_decimal_digits(*v, n);
            }
        }
    }

    Ok(result)
}


fn may_sort_results(e: &Expression, _tss: &[Timeseries]) -> bool {
    return match e {
        Expression::Function(fe) => {
            !fe.function.sorts_results()
        },
        Expression::Aggregation(ae) => {
            !ae.function.sorts_results()
        },
        _ => true
    }
}

pub(crate) fn timeseries_to_result(tss: &mut [Timeseries], may_sort: bool) -> RuntimeResult<Vec<QueryResult>> {
    let mut result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let mut m: HashSet<String> = HashSet::with_capacity(tss.len());

    let timestamps = tss[0].timestamps.deref();

    for ts in tss.iter() {
        if ts.is_all_nans() {
            continue;
        }

        let key = ts.metric_name.to_string();

        if m.contains(&key) {
            return Err(RuntimeError::from(format!("duplicate output timeseries: {}", ts.metric_name)));
        }
        m.insert(key);
        let res = QueryResult {
            metric_name: ts.metric_name.clone(),  // todo(perf) into vs clone/take
            values: ts.values.clone(), // todo(perf) .into vs clone/take
            timestamps: timestamps.clone(), // todo: declare field as Rc<Vec<i64>>
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        };

        result.push(res);
    }

    if may_sort {
        result.sort_by(|a, b| {
            a.metric_name.partial_cmp(&b.metric_name).unwrap()
        })
    }

    Ok(result)
}

pub(crate) fn escape_dots(s: &str) -> String {

    let dots_count =  s.matches('.').count();
    if dots_count == 0 {
        return s.to_string()
    }

    let should_escape = |s: &str, pos: usize| -> bool {
        let len = s.len();
        return if pos + 1 == len || pos + 1 < len {
            let s = &s[pos + 1..];
            let ch = s.chars().next().unwrap();
            ch != '*' && ch != '+' && ch != '{'
        } else {
            false
        }
    };

    let mut result = String::with_capacity(s.len() + 2 * dots_count);
    let mut prev_ch: char = '\0';
    for (i, ch) in s.chars().enumerate() {
        if ch == '.' && (i == 0 || prev_ch != '\\') && should_escape(s, i) {
            // Escape a dot if the following conditions are met:
            // - if it isn't escaped already, i.e. if there is no `\` char before the dot.
            // - if there is no regexp modifiers such as '+', '*' or '{' after the dot.
            result.push('\\');
            result.push('.')
        } else {
            result.push(ch);
        }
        prev_ch = ch;
    }

    result
}

pub(crate) fn escape_dots_in_regexp_label_filters(e: &mut Expression) {
    visit_all(e, |expr: &mut Expression| {
        match expr {
            Expression::MetricExpression(me) => {
                for f in me.label_filters.iter_mut() {
                    if f.is_regexp() {
                        f.value = escape_dots(&f.value);
                    }
                }
            }
            _ => {}
        }
    })
}


#[inline]
pub(super) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| {
        !ts.values.iter().all(|v| v.is_nan())
    });
}
