use std::fmt::Display;
use std::sync::Arc;

use tracing::{trace, trace_span, Span};

use metricsql::ast::{
    AggregationExpr, BinaryExpr, DurationExpr, Expr, FunctionExpr, MetricExpr, ParensExpr,
    RollupExpr,
};
use metricsql::functions::{RollupFunction, TransformFunction};
use metricsql::prelude::BuiltinFunction;

use crate::eval::aggregate::{get_timeseries_limit, try_get_arg_rollup_func_with_metric_expr};
use crate::eval::binary::eval_string_op;
use crate::eval::binary::scalar_operations::scalar_binary_operations;
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

pub fn eval_expr(ctx: &Arc<Context>, ec: &EvalConfig, e: &Expr) -> RuntimeResult<QueryValue> {
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
        Expr::BinaryOperator(be) => {
            if tracing {
                let msg = format!("binary op {}", be.op);
                trace!(msg.to_string());
            }
            trace_span!("binary op {}", be.op);
            let rv = eval_binary_op(ctx, ec, be)?;
            trace!("series={}", rv.len());
            Ok(rv)
        }
        Expr::Parens(pe) => {
            trace!("parens");
            let rv = eval_parens_op(ctx, ec, pe)?;
            trace!("series={}", rv.len());
            Ok(rv)
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
        Expr::Function(fe) => eval_function_op(ctx, ec, fe),
        _ => unimplemented!(),
    }
}

fn eval_function_op(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    fe: &FunctionExpr,
) -> RuntimeResult<QueryValue> {
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
            let rf = nrf(&args)?;
            eval_rollup_func(ctx, ec, rf, rf, &e, &re, None).map_err(|err| map_error(err, &e))
        }
        _ => {
            return Err(RuntimeError::NotImplemented(name.to_string()));
        }
    }
}

fn eval_parens_op(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    pe: &ParensExpr,
) -> RuntimeResult<QueryValue> {
    if pe.expressions.is_empty() {
        // should not happen !!
        return Err(RuntimeError::Internal(
            "BUG: empty parens expression".to_string(),
        ));
    }
    if pe.expressions.len() == 1 {
        return eval_expr(ctx, ec, &pe.expressions[0]);
    }
    let union = BuiltinFunction::Transform(TransformFunction::Union);
    let fe = FunctionExpr {
        name: "union".to_string(),
        function: union,
        args: pe.expressions.clone(), // how to avoid ?
        arg_idx_for_optimization: None,
        keep_metric_names: false,
        is_scalar: false,
        return_type: Default::default(),
    };
    let rv = eval_transform_func(ctx, ec, &fe, TransformFunction::Union)?;
    let val = QueryValue::InstantVector(rv);
    Ok(val)
}

fn eval_binary_op(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    be: &BinaryExpr,
) -> RuntimeResult<QueryValue> {
    // todo: tokio.join!() ?
    let lhs = eval_expr(ctx, ec, &be.left)?;
    let rhs = eval_expr(ctx, ec, &be.right)?;
    match (&lhs, &rhs) {
        (Value::Scalar(left), Value::Scalar(right)) => {
            // todo: add support for bool modifier
            let value = scalar_binary_operations(be.op, *left, *right)?;
            Ok(Value::Scalar(value))
        }
        (Value::InstantVector(left), Value::InstantVector(right)) => {
            binaries::vector_bin_op(expr, &left, &right)?
        }
        (Value::InstantVector(left), Value::Scalar(right)) => {
            binaries::vector_scalar_bin_op(expr, &left, right).await?
        }
        (Value::Scalar(left), Value::InstantVector(right)) => {
            binaries::vector_scalar_bin_op(expr, &right, left).await?
        }
        (Value::String(left), QueryValue::String(right)) => {
            let value = eval_string_op(be.op, &left, &right, be.bool_modifier)?;
            Ok(value)
        }
        _ => {
            return Err(RuntimeError::NotImplemented(format!(
                "invalid binary operation: {} {} {}",
                lhs.data_type_name(),
                be.op,
                rhs.data_type_name()
            )));
        }
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
) -> RuntimeResult<QueryValue> {
    let span = if ctx.trace_enabled() {
        // done this way to avoid possible string alloc in the case where
        // logging is disabled
        let name = &ae.name;
        trace_span!("aggregate", name, series = field::Empty)
    } else {
        Span::none()
    }
    .entered();

    if let Ok(handler) = Handler::try_from(ae.function) {
        if let Some(fe) = try_get_arg_rollup_func_with_metric_expr(ae)? {
            // There is an optimized path for calculating `Expression::AggrFuncExpr` over: RollupFunc
            // over Expression::MetricExpr.
            // The optimized path saves RAM for aggregates over big number of time series.
            let (args, re) = compile_rollup_func_args(&fe)?;
            let expr = Expr::Aggregation(ae.clone());
            let func = get_rollup_function(&fe)?;

            let mut res = RollupEvaluator::create_internal(func, &re, expr, args)?;
            res.is_incr_aggregate = true;
            res.incremental_aggr_handler = Some(handler);

            return Ok(ExprEvaluator::Rollup(res));
        }
    }

    let args = eval_exprs_in_parallel(ctx, ec, &ae.args)?;
    let mut afa = AggrFuncArg {
        args,
        ec,
        modifier: ae.modifier, // todo: avoid clone
        limit: get_timeseries_limit(ae)?,
    };

    match exec_aggregate_fn(ae.function, &mut afa) {
        Ok(res) => {
            span.record("series", res.len());
            Ok(QueryValue::InstantVector(res))
        }
        Err(e) => {
            let res = format!("cannot evaluate {}: {:?}", ae, e);
            Err(RuntimeError::General(res))
        }
    }
}

fn eval_exprs_sequentially(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    args: &Vec<Expr>,
) -> RuntimeResult<Vec<Value>> {
    let values: RuntimeResult<Vec<Value>> = args.map(|expr| eval_expr(ctx, ec, expr)).collect();
    values
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
) -> RuntimeResult<(Vec<Value>, RollupExpr, usize)> {
    let mut re: RollupExpr;
    // todo: i dont think we can have a empty arg_idx_for_optimization
    let rollup_arg_idx = fe.arg_idx_for_optimization;

    if rollup_arg_idx.is_some() {
        let rollup_arg_idx = rollup_arg_idx.unwrap();
        if fe.args.len() <= rollup_arg_idx {
            let msg = format!(
                "expecting at least {} args to {}; got {} args; expr: {}",
                rollup_arg_idx + 1,
                fe.name,
                fe.args.len(),
                fe
            );
            return Err(RuntimeResult::General(msg));
        }
    }

    let rollup_arg_idx = rollup_arg_idx.unwrap_or(fe.args.len());

    let mut args = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg)?;
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

    return Ok((args, re, rollup_arg_idx));
}

fn get_rollup_expr_arg(arg: &Expr) -> RuntimeResult<RollupExpr> {
    let mut re: RollupExpr = match arg {
        Expr::Rollup(re) => re.clone(),
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            RollupExpr::new(arg.clone())
        }
    };

    if !re.for_subquery() {
        // Return standard rollup if it doesn't contain subquery.
        return Ok(re);
    }

    return match &re.expr {
        Expr::MetricExpression(me) => {
            // Convert me[w:step] -> default_rollup(me)[w:step]

            // TODO: avoid clone below. Can we just do an into() ?
            let arg = Expr::Rollup(RollupExpr::new(Expr::MetricExpression(me.clone())));

            match FunctionExpr::default_rollup(arg) {
                Err(e) => return Err(RuntimeError::General(format!("{:?}", e))),
                Ok(fe) => {
                    re.expr = Box::new(Expr::Function(fe));
                    Ok(re)
                }
            }
        }
        _ => {
            // arg contains subquery.
            Ok(re)
        }
    };
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
        d.value(step)
    } else {
        0
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
