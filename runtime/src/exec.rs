use std::collections::HashSet;
use std::ops::Deref;
use std::sync::Arc;
use std::string::String;

use chrono::Utc;

use lib::{round_to_decimal_digits};
use metricsql::ast::Expression;

use crate::context::Context;
use crate::eval::{EvalConfig, Evaluator};
use crate::parser_cache::ParseCacheValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::QueryResult;
use crate::timeseries::Timeseries;

pub fn parse_promql_with_cache(ctx: &mut Context, q: &str) -> RuntimeResult<Arc<ParseCacheValue>> {
    ctx.parse_promql(q)
}

/// executes q for the given config.
pub fn exec(context: &Context,
            ec: &mut EvalConfig,
            q: &str,
            is_first_point_only: bool) -> RuntimeResult<Vec<QueryResult>> {

    let start_time = Utc::now();
    if context.config.stats_enabled && context.query_stats.is_enabled() {
        defer! {
            context.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = context.parse_promql(q)?;
    match (&parsed.evaluator, &parsed.expr, &parsed.has_subquery) {
        (Some(evaluator), Some(expr), has_subquery) => {

            if *has_subquery {
                ec.ensure_timestamps()?;
            }

            let ctx = Arc::new(context);
            let qid = context.active_queries.register_with_start(ec, q, start_time.timestamp());
            let rv = evaluator.eval(&ctx, ec);
            context.active_queries.remove(qid);

            if let Err(e) = rv {
                return Err(e);
            }

            let mut rv = rv.unwrap().into_instant_vector(ec)?;
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

#[inline]
pub(super) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| {
        !ts.values.iter().all(|v| v.is_nan())
    });
}
