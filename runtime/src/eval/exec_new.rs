use std::fmt::Display;
use std::sync::Arc;

use metricsql::ast::{AggregationExpr, DurationExpr, Expr, FunctionExpr, MetricExpr, RollupExpr};
use metricsql::functions::{RollupFunction, TransformFunction};
use metricsql::prelude::BuiltinFunction;
use tracing::{trace, trace_span};

use crate::eval::aggregate::{get_timeseries_limit, try_get_arg_rollup_func_with_metric_expr};
use crate::eval::rollup::compile_rollup_func_args;
use crate::functions::aggregate::{
    exec_aggregate_fn, AggrFuncArg, Handler, IncrementalAggrFuncContext,
};
use crate::functions::rollup::{get_rollup_function_factory, rollup_default, RollupFunc};
use crate::functions::transform::{get_transform_func, TransformFuncArg};
use crate::{Context, EvalConfig, QueryValue, RuntimeError, RuntimeResult, Timeseries};

/// The minimum number of points per timeseries for enabling time rounding.
/// This improves cache hit ratio for frequently requested queries over
/// big time ranges.
const MIN_TIMESERIES_POINTS_FOR_TIME_ROUNDING: i32 = 50;

type Value = QueryValue;

/// QueryStats contains various stats for the query.
pub struct QueryStats {
    // series_fetched contains the number of series fetched from storage during the query evaluation.
    pub series_fetched: usize,
}

impl QueryStats {
    pub fn add_series_fetched(&mut self, n: usize) {
        self.series_fetched += n;
    }
}

fn map_error<E: Display>(err: RuntimeError, e: E) -> RuntimeError {
    RuntimeError::General(format!("cannot evaluate {e}: {}", err))
}

pub fn eval_expr(ctx: &Arc<Context>, ec: &EvalConfig, e: &Expr) -> RuntimeResult<Vec<Timeseries>> {
    let is_tracing = ctx.trace_enabled();
    if is_tracing {
        let query = e.to_string();
        query = bytesutil.LimitStringLen(query, 300);
        let may_cache = ec.may_cache();
        trace_span!(
            "eval: query={query}, timeRange={}, step={}, may_cache={may_cache}",
            ec.time_range_string(),
            ec.step
        )
    }
    let rv = eval_expr_internal(ctx, ec, e)?;
    if is_tracing {
        let series_count = rv.len();
        let mut points_per_series = 0;
        if rv.is_empty() {
            points_per_series = rv[0].len();
        }
        let points_count = series_count * points_per_series;
        trace!(
            "series={series_count}, points={points_count}, points_per_series={points_per_series}"
        )
    }
    return Ok(rv);
}

fn eval_expr_internal(ctx: &Arc<Context>, ec: &EvalConfig, e: &Expr) -> RuntimeResult<QueryValue> {
    let tracing = ctx.trace_enabled();
    match e {
        Expr::StringLiteral(s) => Ok(QueryValue::String(s.to_string())),
        Expr::Number(n) => Ok(QueryValue::Scalar(n.value)),
        Expr::Duration(de) => {
            let d = de.value(ec.step);
            let d_sec = d as f64 / 1000_f64;
            Ok(QueryValue::Scalar(d_sec))
        }
        Expr::MetricExpression(me) => {
            let re = RollupExpr::new(me);
            let val = eval_rollup_func(
                ctx,
                ec,
                RollupFunction::DefaultRollup,
                rollup_default,
                &e,
                &re,
                None,
            )
            .map_err(|err| map_error(err, &e))?;
            Ok(val)
        }
        Expr::Rollup(re) => {
            let val = eval_rollup_func(
                ctx,
                ec,
                RollupFunction::DefaultRollup,
                rollup_default,
                &e,
                &re,
                None,
            )
            .map_err(|err| map_error(err, re))?;
            Ok(val)
        }
        Expr::Aggregation(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(ctx, ec, ae).map_err(|err| map_error(err, ae))?;
            trace!("series={}", rv.len());
            Ok(rv)
        }
        Expr::Function(fe) => {
            let name = fe.function.name();
            match fe.function {
                BuiltinFunction::Transform(tf) => {
                    trace_span!("transform {}()", name);
                    let rv = eval_transform_func(ctx, ec, &fe, tf)?;
                    trace!("series={}", rv.len());
                    return Ok(rv);
                }
                BuiltinFunction::Rollup(rf) => {
                    let nrf = get_rollup_function_factory(rf);
                    let (args, re) = eval_rollup_func_args(ctx, ec, &fe)?;
                    let rf = nrf(args)?;
                    eval_rollup_func(ctx, ec, rf, rf, &e, &re, None)
                        .map_err(|err| map_error(err, &e))
                }
                _ => {
                    return Err(RuntimeError::UnsupportedFunction(name.to_string()));
                }
            }
        }
        _ => unimplemented!(),
    }
}

fn eval_transform_func(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    fe: &FunctionExpr,
    func: TransformFunction,
) -> RuntimeResult<Vec<Timeseries>> {
    let handler = get_transform_func(func);
    let args = if func == TransformFunction::Union {
        eval_exprs_in_parallel(ctx, ec, &fe.args)?
    } else {
        eval_exprs_sequentially(ctx, ec, &fe.args)?
    };
    let mut tfa = TransformFuncArg {
        ec,
        fe,
        args,
        keep_metric_names: func.keep_metric_name(),
    };
    handler(&mut tfa).map_err(|err| map_error(err, fe))
}

fn eval_aggr_func(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    ae: &AggregationExpr,
) -> RuntimeResult<Vec<Timeseries>> {
    if let Ok(handler) = Handler::try_from(ae.function) {
        if let Some(fe) = try_get_arg_rollup_func_with_metric_expr(ae)? {
            // There is an optimized path for calculating `Expression::AggrFuncExpr` over: RollupFunc
            // over Expression::MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = compile_rollup_func_args(&fe)?;
            let expr = Expr::Aggregation(ae.clone());
            let func = get_rollup_function(&fe)?;

            let mut res = RollupEvaluator::create_internal(func, &re, expr, args)?;

            res.timeseries_limit = get_timeseries_limit(ae)?;
            res.is_incr_aggregate = true;
            res.incremental_aggr_handler = Some(handler);

            return Ok(ExprEvaluator::Rollup(res));
        }
    }
    if let Some(callbacks) = getIncrementalAggrFuncCallbacks(ae.name) {
        let (fe, nrf) = try_get_arg_rollup_func_with_metric_expr(ae);
        if let Some(fe) = fe {
            // There is an optimized path for calculating AggrFuncExpr over rollupFunc over MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = eval_rollup_func_args(ctx, ec, fe)?;
            let rf = nrf(args);
            let iafc = newIncrementalAggrFuncContext(ae, callbacks);
            return eval_rollup_func(ctx, ec, fe.Name, rf, ae, re, iafc);
        }
    }
    let args = eval_exprs_in_parallel(ctx, ec, &ae.args)?;
    let mut afa = AggrFuncArg {
        args,
        ec,
        modifier: None,
        limit: get_timeseries_limit(ae)?,
    };

    trace!("eval {}", ae.name);
    exec_aggregate_fn(ae.function, &mut afa).map_err(|err| map_error(err, ae))
}

fn eval_exprs_sequentially(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    args: &[Expr],
) -> RuntimeResult<Vec<Value>> {
    todo!("eval_exprs_sequentially")
}

fn eval_exprs_in_parallel(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    args: &[Expr],
) -> RuntimeResult<Vec<Value>> {
    todo!("eval_exprs_in_parallel")
}

pub(super) fn eval_rollup_func_args(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    fe: &FunctionExpr,
) -> RuntimeResult<(Vec<Value>, RollupExpr)> {
    let mut re: RollupExpr;

    let rollup_arg_idx = fe.get_rollup_arg_idx();
    if fe.args.len() <= rollup_arg_idx {
        let msg = format!(
            "expecting at least {} args to {}; got {} args; expr: {}",
            rollup_arg_idx + 1,
            fe.name(),
            fe.args.len,
            fe
        );
        return Err(RuntimeResult::General(msg));
    }

    let args = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg);
            args.push(re);
            continue;
        }
        let ts = eval_expr(ctx, ec, arg).map_err(|err| {
            Err(RuntimeError::General(format!(
                "cannot evaluate arg #{} for {}: {}",
                i + 1,
                fe,
                err
            )))
        })?;

        args.push(ts);
    }

    return Ok((args, re));
}

// expr may contain:
// - rollupFunc(m) if iafc is None
// - aggrFunc(rollupFunc(m)) if iafc isn't None
fn eval_rollup_func(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    function: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: Option<&IncrementalAggrFuncContext>,
) -> RuntimeResult<QueryValue> {
    if re.at.is_none() {
        return eval_rollup_func_without_at(ctx, ec, function, rf, expr, re, iafc);
    }

    let at_expr = re.at.as_ref().unwrap();
    let at_timestamp = get_at_timestamp(ctx, ec, &at_expr)?;

    let mut ec_new = ec.clone();
    ec_new.start = at_timestamp;
    ec_new.end = at_timestamp;
    eval_rollup_func_without_at(ctx, &ec_new, function, rf, expr, re, iafc)
}

pub fn eval_rollup_func_without_at(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
    iafc: Option<&IncrementalAggrFuncContext>,
) -> RuntimeResult<QueryValue> {
    todo!("eval_rollup_func_without_at")
}

fn get_duration(dur: &Option<DurationExpr>, step: i64) -> i64 {
    if let Some(d) = dur {
        d.value(d, step)
    } else {
        0
    }
}

fn get_at_timestamp(ctx: &Arc<Context>, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<i64> {
    match eval_expr(ctx, ec, expr) {
        Err(err) => {
            let msg = format!("cannot evaluate `@` modifier: {:?}", err);
            return Err(RuntimeError::from(msg));
        }
        Ok(tss_at) => {
            match tss_at {
                QueryValue::Scalar(v) => Ok((v * 1000_f64) as i64),
                QueryValue::InstantVector(v) => {
                    if v.len() != 1 {
                        let msg = format!("`@` modifier must return a single series; it returns {} series instead", v.len());
                        return Err(RuntimeError::from(msg));
                    }
                    let ts = &v[0];
                    if ts.values.len() > 1 {
                        let msg = format!(
                            "`@` modifier must return a single value; it returns {} series instead",
                            ts.values.len()
                        );
                        return Err(RuntimeError::from(msg));
                    }
                    Ok((ts.values[0] * 1000_f64) as i64)
                }
                _ => {
                    let val = tss_at.get_int()?;
                    Ok(val * 1000_i64)
                }
            }
        }
    }
}

fn eval_rollup_func_with_subquery(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    re: &RollupExpr,
) -> RuntimeResult<Vec<Timeseries>> {
    todo!("eval_rollup_func_with_subquery")
}

fn eval_rollup_func_with_metric_expr(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    func: RollupFunction,
    rf: RollupFunc,
    expr: &Expr,
    me: &MetricExpr,
    iafc: &IncrementalAggrFuncContext,
    window_expr: &DurationExpr,
) -> RuntimeResult<QueryValue> {
    todo!("eval_rollup_func_with_metric_expr")
}
