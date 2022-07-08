use std::ops::Deref;
use metricsql::ast::{AggrFuncExpr, Expression, FuncExpr};
use metricsql::parser::rollup::get_rollup_arg_idx;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::{create_evaluator, create_evaluators, eval_args, ExprEvaluator};
use crate::eval::rollup::{compile_rollup_func_args, RollupEvaluator};
use crate::eval::traits::Evaluator;

use crate::functions::{
    rollup::{get_rollup_func},
    aggr::{AggrFunc, AggrFuncArg, get_aggr_func},
    aggr_incremental::{
        get_incremental_aggr_func_callbacks,
        IncrementalAggrFuncContext
    }
};


pub(super) fn create_aggr_evaluator(ae: &AggrFuncExpr) -> RuntimeResult<ExprEvaluator> {
    match get_incremental_aggr_func_callbacks(&ae.name) {
        Some(callbacks) => {
            match try_get_arg_rollup_func_with_metric_expr(ae) {
                Some(fe) => {
                    // There is an optimized path for calculating `Expression::AggrFuncExpr` over: RollupFunc
                    // over Expression::MetricExpr.
                    // The optimized path saves RAM for aggregates over big number of time series.
                    let (args, re, rollup_index) = compile_rollup_func_args(&fe)?;
                    let iafc = IncrementalAggrFuncContext::new(ae, callbacks);
                    let expr = Expression::Aggregation(ae.clone());

                    let mut res = RollupEvaluator::create_internal(
                        &ae.name,
                        &re,
                        expr,
                        args
                    )?;
                    res.iafc = Some(iafc);
                    res.rollup_index = rollup_index as i32;
                    res.evaluator = Box::new( create_evaluator(&expr)? );
                    return Ok(ExprEvaluator::Rollup(res))
                },
                _ => {}
            }
        }
        _ => {}
    }

    Ok(ExprEvaluator::Aggregate(AggregateEvaluator::new(ae)?))
}


pub(crate) struct AggregateEvaluator {
    ae: AggrFuncExpr,
    aggr_func: &'static AggrFunc,
    args: Vec<ExprEvaluator>
}

impl AggregateEvaluator {
    pub fn new(ae: &AggrFuncExpr) -> RuntimeResult<Self> {
        match get_aggr_func(&ae.name) {
            None => panic!("Bug: unknown aggregate function {}", ae.name),
            Some(aggr_func) => {
                let args = create_evaluators(&ae.args)?;
                Ok(Self {
                    aggr_func,
                    args,
                    ae: ae.clone()
                })
            }
        }
    }
}

impl Evaluator for AggregateEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let args = eval_args(ctx, ec, &self.args)?;
        //todo: use tinyvec for args
        let mut afa = AggrFuncArg::new(*self.ae, args, ec);
        match (self.aggr_func)(&mut afa) {
            Ok(res) => Ok(res),
            Err(e) => {
                let res = format!("cannot evaluate {}: {:?}", self.ae, e);
                Err(RuntimeError::General(res))
            }
        }
    }
}


pub(super) fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> Option<FuncExpr> {
    if ae.args.len() != 1 {
        return None;
    }
    let e = ae.args[0].deref();
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    return match ae.args[0].deref() {
        Expression::MetricExpression(me) => {
            if me.is_empty() {
                return None;
            }
            let fe = FuncExpr::default_rollup( *e);
            Some(fe)
        }
        Expression::Rollup(re) => {
            let mut is_me: bool = false;
            let mut is_empty: bool = false;
            match re.expr.deref() {
                Expression::MetricExpression(me) => {
                    is_me = true;
                    is_empty = me.is_empty();
                },
                _ => {}
            }
            if !is_me || is_empty || re.for_subquery() {
                return None;
            }
            // e = metricExpr[d]
            let fe = FuncExpr::default_rollup( *e);
            Some(fe)
        }
        Expression::Function(fe) => {
            let nrf = get_rollup_func(&fe.name);
            if nrf.is_none() {
                return None;
            }
            let rollup_arg_idx = get_rollup_arg_idx(fe);
            if rollup_arg_idx >= fe.args.len() as i32 {
                // Incorrect number of args for rollup func.
                return None;
            }
            let arg = fe.args[rollup_arg_idx as usize].deref();
            match arg {
                Expression::MetricExpression(me) => {
                    if me.is_empty() {
                        return None;
                    }
                    // e = rollupFunc(metricExpr)
                    let f = FuncExpr::from_single_arg(&fe.name, *arg);
                    Some(f)
                }
                Expression::Rollup(re) => {
                    match re.expr.deref() {
                        Expression::MetricExpression(me) => {
                            if me.is_empty() || re.for_subquery() {
                                return None;
                            }
                            // e = RollupFunc(metricExpr[d])
                            // todo: use COW to avoid clone
                            Some(fe.clone())
                        },
                        _ => None
                    }
                }
                _ => None
            }
        }
        _ => None
    }
}
