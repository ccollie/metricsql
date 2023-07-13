use std::collections::HashSet;
use std::ops::Deref;
use std::string::String;
use std::sync::Arc;

use chrono::Utc;
use tracing::{field, info, trace_span, Span};

use lib::round_to_decimal_digits;
use metricsql::ast::Expr;

use crate::context::Context;
use crate::eval::{eval_expr, EvalConfig};
use crate::parser_cache::ParseCacheValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::QueryResult;
use crate::types::Timeseries;
use crate::{ParseCacheResult, QueryValue};

pub(crate) fn parse_promql_internal(
    context: &Arc<Context>,
    query: &str,
) -> RuntimeResult<Arc<ParseCacheValue>> {
    let span = trace_span!("parse", cached = field::Empty).entered();
    let (parsed, cached) = context.parse_promql(query)?;
    span.record("cached", cached == ParseCacheResult::CacheHit);
    Ok(parsed)
}

pub(crate) fn exec_internal(
    context: &Arc<Context>,
    ec: &mut EvalConfig,
    q: &str,
) -> RuntimeResult<(QueryValue, Arc<ParseCacheValue>)> {
    let start_time = Utc::now();
    if context.stats_enabled() {
        defer! {
            context.query_stats.register_query(q, ec.end - ec.start, start_time)
        }
    }

    ec.validate()?;

    let parsed = parse_promql_internal(context, q)?;

    match (&parsed.expr, &parsed.has_subquery) {
        (Some(expr), has_subquery) => {
            if *has_subquery {
                ec.ensure_timestamps()?;
            }

            let ctx = Arc::new(context);
            let qid = context
                .active_queries
                .register(ec, q, Some(start_time.timestamp()));

            defer! {
                context.active_queries.remove(qid);
            }

            let is_tracing = ctx.trace_enabled();

            let span = if is_tracing {
                let mut query = q.to_string();
                query.truncate(300);

                trace_span!(
                    "eval",
                    query,
                    may_cache = ec.may_cache(),
                    start = ec.start,
                    end = ec.end,
                    series = field::Empty,
                    points = field::Empty,
                    points_per_series = field::Empty
                )
            } else {
                Span::none()
            }
            .entered();

            let rv = eval_expr(&ctx, ec, expr)?;

            if is_tracing {
                let ts_count: usize;
                let series_count: usize;
                match &rv {
                    QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => {
                        series_count = iv.len();
                        if series_count > 0 {
                            ts_count = iv[0].timestamps.len();
                        } else {
                            ts_count = 0;
                        }
                    }
                    _ => {
                        ts_count = ec.timestamps().len();
                        series_count = 1;
                    }
                }
                let mut points_per_series = 0;
                if series_count > 0 {
                    points_per_series = ts_count
                }

                let points_count = series_count * points_per_series;
                span.record("series", series_count);
                span.record("points", points_count);
                span.record("points_per_series", points_per_series);
            }

            Ok((rv, Arc::clone(&parsed)))
        }
        _ => {
            panic!("Bug: Invalid parse state")
        }
    }
}

/// executes q for the given config.
pub fn exec(
    context: &Arc<Context>,
    ec: &mut EvalConfig,
    q: &str,
    is_first_point_only: bool,
) -> RuntimeResult<Vec<QueryResult>> {
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

    // at this point, parsed.evaluator is Some, but lets avoid unwrap in any case
    let may_sort = if let Some(expr) = &parsed.expr {
        may_sort_results(expr, &rv)
    } else {
        false
    };

    let mut result = timeseries_to_result(&mut rv, may_sort)?;

    let n = ec.round_digits as u8;
    if n < 100 {
        for r in result.iter_mut() {
            for v in r.values.iter_mut() {
                *v = round_to_decimal_digits(*v, n as i16);
            }
        }
    }

    info!("sorted = {may_sort}, round_digits = {}", ec.round_digits);

    Ok(result)
}

fn may_sort_results(e: &Expr, _tss: &[Timeseries]) -> bool {
    return match e {
        Expr::Function(fe) => !fe.function.may_sort_results(),
        Expr::Aggregation(ae) => !ae.function.may_sort_results(),
        _ => true,
    };
}

pub(crate) fn timeseries_to_result(
    tss: &mut [Timeseries],
    may_sort: bool,
) -> RuntimeResult<Vec<QueryResult>> {
    let mut result: Vec<QueryResult> = Vec::with_capacity(tss.len());
    let mut m: HashSet<String> = HashSet::with_capacity(tss.len());

    let timestamps = tss[0].timestamps.deref();

    for ts in tss.iter() {
        if ts.is_all_nans() {
            continue;
        }

        // todo: use hash
        let key = ts.metric_name.to_string();

        if m.contains(&key) {
            return Err(RuntimeError::from(format!(
                "duplicate output timeseries: {}",
                ts.metric_name
            )));
        }
        m.insert(key);
        let res = QueryResult {
            metric_name: ts.metric_name.clone(), // todo(perf) into vs clone/take
            values: ts.values.clone(),           // todo(perf) .into vs clone/take
            timestamps: timestamps.clone(),      // todo: declare field as Rc<Vec<i64>>
            rows_processed: 0,
            worker_id: 0,
        };

        result.push(res);
    }

    if may_sort {
        result.sort_by(|a, b| a.metric_name.partial_cmp(&b.metric_name).unwrap())
    }

    Ok(result)
}

#[inline]
pub(super) fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| !ts.values.iter().all(|v| v.is_nan()));
}
