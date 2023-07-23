use std::fmt::Display;
use std::sync::Arc;

use tracing::{field, trace, trace_span, Span};

use metricsql::ast::{BinaryExpr, Expr, FunctionExpr, ParensExpr, RollupExpr};
use metricsql::functions::{RollupFunction, TransformFunction};
use metricsql::prelude::BuiltinFunction;

use crate::eval::aggregate::eval_aggr_func;
use crate::eval::binary::{
    eval_duration_duration_op, eval_duration_scalar_op, eval_scalar_vector_binop,
    eval_string_string_op, eval_vector_scalar_binop, eval_vector_vector_binop,
    scalar_binary_operations,
};
use crate::eval::rollups::RollupExecutor;
use crate::functions::rollup::{get_rollup_function_factory, rollup_default, RollupHandlerEnum};
use crate::functions::transform::{get_transform_func, TransformFuncArg};
use crate::{Context, EvalConfig, QueryValue, RuntimeError, RuntimeResult, Timeseries};

type Value = QueryValue;

fn map_error<E: Display>(err: RuntimeError, e: E) -> RuntimeError {
    RuntimeError::General(format!("cannot evaluate {e}: {}", err))
}

pub fn exec_expr(ctx: &Arc<Context>, ec: &EvalConfig, expr: &Expr) -> RuntimeResult<QueryValue> {
    let tracing = ctx.trace_enabled();
    match expr {
        Expr::StringLiteral(s) => Ok(QueryValue::String(s.to_string())),
        Expr::Number(n) => Ok(QueryValue::Scalar(n.value)),
        Expr::Duration(de) => {
            let d = de.value(ec.step);
            let d_sec = d as f64 / 1000_f64;
            Ok(QueryValue::Scalar(d_sec))
        }
        Expr::BinaryOperator(be) => {
            let span = if tracing {
                trace_span!("binary op", "op" = be.op.as_str(), series = field::Empty)
            } else {
                Span::none()
            }
            .entered();

            let rv = eval_binary_op(ctx, ec, be)?;

            span.record("series", rv.len());

            Ok(rv)
        }
        Expr::Parens(pe) => {
            trace_span!("parens");
            let rv = eval_parens_op(ctx, ec, pe)?;
            Ok(rv)
        }
        Expr::MetricExpression(_me) => {
            let re = RollupExpr::new(expr.clone());
            let handler = RollupHandlerEnum::Wrapped(rollup_default);
            let mut executor =
                RollupExecutor::new(RollupFunction::DefaultRollup, handler, expr, &re);
            let val = executor
                .eval(ctx, ec)
                .map_err(|err| map_error(err, &expr))?;
            Ok(val)
        }
        Expr::Rollup(re) => {
            let handler = RollupHandlerEnum::Wrapped(rollup_default);
            let mut executor =
                RollupExecutor::new(RollupFunction::DefaultRollup, handler, expr, &re);
            executor.eval(ctx, ec).map_err(|err| map_error(err, &expr))
        }
        Expr::Aggregation(ae) => {
            trace!("aggregate {}()", ae.function.name());
            let rv = eval_aggr_func(ctx, ec, expr, ae).map_err(|err| map_error(err, ae))?;
            trace!("series={}", rv.len());
            Ok(rv)
        }
        Expr::Function(fe) => eval_function_op(ctx, ec, expr, fe),
        _ => unimplemented!(),
    }
}

fn eval_function_op(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    expr: &Expr,
    fe: &FunctionExpr,
) -> RuntimeResult<QueryValue> {
    let name = fe.function.name();
    return match fe.function {
        BuiltinFunction::Transform(tf) => {
            let span = if ctx.trace_enabled() {
                trace_span!("transform", function = fe.name, series = field::Empty)
            } else {
                Span::none()
            }
            .entered();

            let rv = eval_transform_func(ctx, ec, &fe, tf)?;
            span.record("series", rv.len());

            Ok(QueryValue::InstantVector(rv))
        }
        BuiltinFunction::Rollup(rf) => {
            let nrf = get_rollup_function_factory(rf);
            let (args, re, _) = eval_rollup_func_args(ctx, ec, &fe)?;
            let func_handler = nrf(&args, ec)?;
            let mut rollup_handler = RollupExecutor::new(rf, func_handler, expr, &re);
            let val = rollup_handler
                .eval(ctx, ec)
                .map_err(|err| map_error(err, fe))?;
            Ok(val)
        }
        _ => Err(RuntimeError::NotImplemented(name.to_string())),
    };
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
        return exec_expr(ctx, ec, &pe.expressions[0]);
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
    let is_tracing = ctx.trace_enabled();
    let res = match (&be.left.as_ref(), &be.right.as_ref()) {
        // vector op vector needs special handling
        (Expr::MetricExpression(_), Expr::MetricExpression(_)) => {
            eval_vector_vector_binop(be, ctx, ec)
        }
        // the following cases can be handled cheaply without invoking async runtime
        (Expr::Number(left), Expr::Number(right)) => {
            let value = scalar_binary_operations(be.op, left.value, right.value, be.bool_modifier)?;
            Ok(Value::Scalar(value))
        }
        (Expr::Duration(left), Expr::Duration(right)) => {
            eval_duration_duration_op(&left, right, be.op, ec.step)
        }
        (Expr::Duration(dur), Expr::Number(scalar)) => {
            eval_duration_scalar_op(&dur, scalar.value, be.op, ec.step)
        }
        (Expr::Number(scalar), Expr::Duration(dur)) => {
            eval_duration_scalar_op(&dur, scalar.value, be.op, ec.step)
        }
        (Expr::StringLiteral(left), Expr::StringLiteral(right)) => {
            eval_string_string_op(be.op, &left, &right, be.bool_modifier)
        }
        (left, right) => {
            // todo: rayon.join!
            let lhs = exec_expr(ctx, ec, &left)?;
            let rhs = exec_expr(ctx, ec, &right)?;

            match (lhs, rhs) {
                (QueryValue::Scalar(left), QueryValue::Scalar(right)) => {
                    let value = scalar_binary_operations(be.op, left, right, be.bool_modifier)?;
                    Ok(Value::Scalar(value))
                }
                (QueryValue::InstantVector(_), QueryValue::InstantVector(_)) => {
                    eval_vector_vector_binop(be, ctx, ec)
                }
                (QueryValue::InstantVector(vector), QueryValue::Scalar(scalar)) => {
                    eval_vector_scalar_binop(be, vector, scalar, is_tracing)
                }
                (QueryValue::Scalar(scalar), QueryValue::InstantVector(vector)) => {
                    eval_scalar_vector_binop(be, vector, scalar, is_tracing)
                }
                (QueryValue::String(left), QueryValue::String(right)) => {
                    eval_string_string_op(be.op, &left, &right, be.bool_modifier)
                }
                _ => {
                    return Err(RuntimeError::NotImplemented(format!(
                        "invalid binary operation: {} {} {}",
                        be.left.variant_name(),
                        be.op,
                        be.right.variant_name()
                    )));
                }
            }
        }
    };
    res
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

pub(super) fn eval_exprs_sequentially(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    args: &Vec<Expr>,
) -> RuntimeResult<Vec<Value>> {
    let res = args
        .iter()
        .map(|expr| exec_expr(ctx, ec, expr))
        .collect::<RuntimeResult<Vec<Value>>>();

    res
}

pub(super) fn eval_exprs_in_parallel(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    args: &Vec<Expr>,
) -> RuntimeResult<Vec<Value>> {
    eval_exprs_sequentially(ctx, ec, args)
    // todo!("eval_exprs_in_parallel")
}

pub(super) fn eval_rollup_func_args(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    fe: &FunctionExpr,
) -> RuntimeResult<(Vec<Value>, RollupExpr, usize)> {
    let mut re: RollupExpr = Default::default();
    // todo: i dont think we can have a empty arg_idx_for_optimization
    let rollup_arg_idx = fe.arg_idx_for_optimization.expect("rollup_arg_idx is None");

    if fe.args.len() <= rollup_arg_idx {
        let msg = format!(
            "expecting at least {} args to {}; got {} args; expr: {}",
            rollup_arg_idx + 1,
            fe.name,
            fe.args.len(),
            fe
        );
        return Err(RuntimeError::from(msg));
    }

    let mut args = Vec::with_capacity(fe.args.len());
    for (i, arg) in fe.args.iter().enumerate() {
        if i == rollup_arg_idx {
            re = get_rollup_expr_arg(arg)?;
            args.push(QueryValue::Scalar(f64::NAN)); // placeholder
            continue;
        }
        let value = exec_expr(ctx, ec, arg).map_err(|err| {
            let msg = format!("cannot evaluate arg #{} for {}: {}", i + 1, fe, err);
            RuntimeError::ArgumentError(msg)
        })?;

        args.push(value);
    }

    return Ok((args, re, rollup_arg_idx));
}

// todo: COW
fn get_rollup_expr_arg(arg: &Expr) -> RuntimeResult<RollupExpr> {
    return match arg {
        Expr::Rollup(re) => {
            let mut re = re.clone();
            if !re.for_subquery() {
                // Return standard rollup if it doesn't contain subquery.
                return Ok(re);
            }

            match &re.expr.as_ref() {
                Expr::MetricExpression(_) => {
                    // Convert me[w:step] -> default_rollup(me)[w:step]

                    let arg = Expr::Rollup(RollupExpr::new(*re.expr.clone()));

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
            }
        }
        _ => {
            // Wrap non-rollup arg into RollupExpr.
            Ok(RollupExpr::new(arg.clone()))
        }
    };
}
