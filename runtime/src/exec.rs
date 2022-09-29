use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;

use lib::{get_pooled_buffer, round_to_decimal_digits};
use metricsql::ast::Expression;

use crate::context::Context;
use crate::eval::{EvalConfig, Evaluator};
use crate::parser_cache::ParseCacheValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::QueryResult;
use crate::timeseries::Timeseries;

pub fn parse_promql_with_cache(ctx: &mut Context, q: &str) -> RuntimeResult<Arc<ParseCacheValue>> {
    let cached = ctx.parse_cache.parse(q);
    if let Some(err) = &cached.err {
        return Err(RuntimeError::ParseError(err.clone()))
    }
    return Ok(cached)
}

/// executes q for the given config.
pub fn exec(env: &mut Context,
            ec: &mut EvalConfig,
            q: &str,
            is_first_point_only: bool) -> RuntimeResult<Vec<QueryResult>> {

    if env.query_stats.enabled() {
        let start_time = Utc::now();
        defer! {
            env.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = env.parse_cache.parse(q);
    match (&parsed.err, &parsed.evaluator, &parsed.expr, &parsed.has_subquery) {
        (Some(err), None, None, _) => {
            return Err(RuntimeError::ParseError(err.clone()))
        },
        (None, Some(evaluator), Some(expr), has_subquery) => {
            let qid = env.active_queries.add(ec, q);

            if *has_subquery {
                ec.ensure_timestamps()?;
            }

            let rv = evaluator.eval(env, ec);
            match rv {
                Err(e) => {
                    env.active_queries.remove(qid);
                    return Err(e)
                },
                _ => {}
            }

            let mut rv = rv.unwrap();
            if is_first_point_only {
                if rv[0].timestamps.len() > 0 {
                    let timestamps = Arc::new(vec![rv[0].timestamps[0]]);
                    // Remove all the points except the first one from every time series.
                    for ts in rv.iter_mut() {
                        ts.values.resize(1, 0.0);
                        ts.timestamps = Arc::clone(&timestamps);
                    }
                }
            }

            let may_sort = may_sort_results(expr, &rv);
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
        },
        _ => {
            panic!("Bug: Invalid parse state")
        }
    }
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

pub(crate) fn timeseries_to_result(tss: &mut Vec<Timeseries>, may_sort: bool) -> RuntimeResult<Vec<QueryResult>> {
    remove_empty_series(tss);
    let mut result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let mut m: HashSet<String> = HashSet::with_capacity(tss.len());
    let mut bb = get_pooled_buffer(256);

    for (i, ts) in tss.iter_mut().enumerate() {
        let key = ts.metric_name.marshal_to_string(&mut bb);
        if m.contains(key.as_ref()) {
            return Err(RuntimeError::from(format!("duplicate output timeseries: {}", ts.metric_name)));
        }
        m.insert(key.to_string());
        result[i].metric_name.copy_from(&ts.metric_name);
        result[i].values = ts.values.into();
        result[i].timestamps.extend_from_slice(&ts.timestamps);
        bb.clear();
    }

    if may_sort {
        result.sort_by(|a, b| {
            a.metric_name.partial_cmp(&b.metric_name).unwrap()
        })
    }

    Ok(result)
}

#[inline]
pub(super) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| {
        !ts.values.iter().all(|v| v.is_nan())
    });
}
