use std::ops::Deref;
use std::str::FromStr;
use metricsql::ast::{AggregateModifier, AggrFuncExpr, Expression, FuncExpr};
use metricsql::parser::rollup::get_rollup_arg_idx;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use super::{create_evaluators, Evaluator, eval_params, eval_volatility, ExprEvaluator};
use crate::eval::rollup::{compile_rollup_func_args, RollupEvaluator};

use crate::functions::rollup::get_rollup_func;
use crate::functions::{
    aggregate::{
        AggrFn,
        AggrFuncArg,
        AggregateFunction,
        get_aggr_func,
        get_incremental_aggr_func_callbacks,
        IncrementalAggrFuncContext
    },
};

use crate::functions::types::{Signature, Volatility};


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
    expr: String,
    function: AggregateFunction,
    signature: Signature,
    handler: Box<dyn AggrFn + 'static>,
    args: Vec<ExprEvaluator>,
    /// optional modifier such as `by (...)` or `without (...)`.
    modifier: Option<AggregateModifier>,
    limit: usize
}

impl AggregateEvaluator {
    pub fn new(ae: &AggrFuncExpr) -> RuntimeResult<Self> {
        let handler = get_aggr_func(&ae.name)?;
        let function = AggregateFunction::from_str(&ae.name)?;
        let signature = function.signature();
        let args = create_evaluators(&ae.args)?;
        let limit = ae.limit;

        signature.validate_arg_count(&ae.name, args.len())?;

        Ok(Self {
            handler: Box::new(handler),
            args,
            function,
            signature,
            modifier: ae.modifier.clone(),
            limit,
            expr: ae.to_string()
        })
    }
}

impl Evaluator for AggregateEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let args = eval_params(ctx, ec, &self.signature.type_signature, &self.args)?;
        //todo: use tinyvec for args
        let mut afa = AggrFuncArg::new(ec, args,  &self.modifier, self.limit);
        match (self.handler)(&mut afa) {
            Ok(res) => Ok(res),
            Err(e) => {
                let res = format!("cannot evaluate {}: {:?}", self.expr, e);
                Err(RuntimeError::General(res))
            }
        }
    }

    fn volatility(&self) -> Volatility { 
        eval_volatility(&self.signature, &self.args)
    }
}


fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> Option<FuncExpr> {
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
            let fe = FuncExpr::default_rollup( e.clone());
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
            let fe = FuncExpr::default_rollup( e.clone());
            Some(fe)
        }
        Expression::Function(fe) => {
            let nrf = get_rollup_func(&fe.name);
            if nrf.is_err() {
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
                    let f = FuncExpr::from_single_arg(&fe.name, arg.clone());
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
