use std::ops::Deref;
use std::str::FromStr;
use std::sync::Arc;

use metricsql::ast::{AggregateModifier, AggrFuncExpr, Expression, FuncExpr, MetricExpr};
use metricsql::functions::{AggregateFunction, BuiltinFunction, RollupFunction, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::arg_list::ArgList;
use crate::eval::rollup::{compile_rollup_func_args, IncrementalAggrFuncOptions, RollupEvaluator};
use crate::functions::{
    aggregate::{
        AggrFn,
        AggrFuncArg,
        get_incremental_aggr_func_callbacks
    },
};
use crate::functions::aggregate::get_aggr_func;
use crate::functions::types::AnyValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::utils::num_cpus;

use super::{Evaluator, ExprEvaluator};

pub(super) fn create_aggr_evaluator(ae: &AggrFuncExpr) -> RuntimeResult<ExprEvaluator> {
    match get_incremental_aggr_func_callbacks(&ae.name) {
        Some(callbacks) => {
            match try_get_arg_rollup_func_with_metric_expr(ae)? {
                Some(fe) => {
                    // There is an optimized path for calculating `Expression::AggrFuncExpr` over: RollupFunc
                    // over Expression::MetricExpr.
                    // The optimized path saves RAM for aggregates over big number of time series.
                    let (args, re) = compile_rollup_func_args(&fe)?;
                    let expr = Expression::Aggregation(ae.clone());
                    let func = get_rollup_function(&fe)?;

                    let mut res = RollupEvaluator::create_internal(
                        func,
                        &re,
                        expr,
                        args
                    )?;

                    res.timeseries_limit = get_timeseries_limit(ae)?;
                    res.is_incr_aggregate = true;
                    res.incremental_aggr_opts = Some(IncrementalAggrFuncOptions {
                        callbacks,
                    });

                    return Ok(ExprEvaluator::Rollup(res))
                },
                _ => {}
            }
        }
        _ => {}
    }

    Ok(ExprEvaluator::Aggregate(AggregateEvaluator::new(ae)?))
}


pub struct AggregateEvaluator {
    pub expr: String,
    pub function: AggregateFunction,
    args: ArgList,
     /// optional modifier such as `by (...)` or `without (...)`.
    modifier: Option<AggregateModifier>,
    /// Max number of timeseries to return
    pub limit: usize,
    handler: Arc<dyn AggrFn + 'static>,
}

impl AggregateEvaluator {
    pub fn new(ae: &AggrFuncExpr) -> RuntimeResult<Self> {
        // todo: remove unwrap and return a Result
        let function = AggregateFunction::from_str(&ae.name).unwrap();
        let handler = get_aggr_func(&function)?;
        let signature = function.signature();
        let args = ArgList::new(&signature, &ae.args)?;

        Ok(Self {
            handler,
            args,
            function,
            modifier: ae.modifier.clone(),
            limit: ae.limit,
            expr: ae.to_string()
        })
    }

    pub fn is_idempotent(&self) -> bool {
        self.volatility() != Volatility::Volatile && self.args.all_const()
    }
}

impl Evaluator for AggregateEvaluator {
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        let args = self.args.eval(ctx, ec)?;
        //todo: use tinyvec for args
        let mut afa = AggrFuncArg::new(ec, args,  &self.modifier, self.limit);
        match (self.handler)(&mut afa) {
            Ok(res) => Ok(AnyValue::InstantVector(res)),
            Err(e) => {
                let res = format!("cannot evaluate {}: {:?}", self.expr, e);
                Err(RuntimeError::General(res))
            }
        }
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }
}


fn try_get_arg_rollup_func_with_metric_expr(ae: &AggrFuncExpr) -> RuntimeResult<Option<FuncExpr>> {
    if ae.args.len() != 1 {
        return Ok(None);
    }
    let e = ae.args[0].deref();
    // Make sure e contains one of the following:
    // - metricExpr
    // - metricExpr[d]
    // -: RollupFunc(metricExpr)
    // -: RollupFunc(metricExpr[d])

    fn create_func(me: &MetricExpr, expr: &Expression, name: &str, for_subquery: bool) -> RuntimeResult<Option<FuncExpr>> {
        if me.is_empty() || for_subquery {
            return Ok(None)
        }

        let func_name = if name.len() == 0 {
            "default_rollup"
        } else {
            name
        };

        let span = me.span.clone();
        match FuncExpr::from_single_arg(func_name, expr.clone(), span) {
            Err(e) => {
                Err(RuntimeError::General(
                    format!("Error creating function {}: {:?}", func_name, e)
                ))
            },
            Ok(fe) => Ok(Some(fe))
        }
    }


   return match ae.args[0].deref() {
        Expression::MetricExpression(me) => {
            return create_func(me, e, "", false)
        }
        Expression::Rollup(re) => {
            match re.expr.deref() {
                Expression::MetricExpression(me) => {
                    // e = metricExpr[d]
                    create_func(me, e, "", re.for_subquery())
                },
                _ => Ok(None)
            }
        }
        Expression::Function(fe) => {
            let function = BuiltinFunction::from_str(&fe.name);
            match function {
                BuiltinFunction::Rollup(_) => {
                    match fe.get_arg_for_optimization() {
                        None => {
                            // Incorrect number of args for rollup func.
                            // TODO: this should be an error
                            // all rollup functions should have a value for this
                            return Ok(None);
                        },
                        Some(arg) => {
                            match arg.deref() {
                                Expression::MetricExpression(me) => {
                                    create_func(me, e, &fe.name, false)
                                },
                                Expression::Rollup(re) => {
                                    match &*re.expr {
                                        Expression::MetricExpression(me) => {
                                            if me.is_empty() || re.for_subquery() {
                                                Ok(None)
                                            } else {
                                                // e = RollupFunc(metricExpr[d])
                                                // todo: use COW to avoid clone
                                                Ok(Some(fe.clone()))
                                            }
                                        },
                                        _ => Ok(None)
                                    }

                                },
                                _=> Ok(None)
                            }
                        }

                    }
                }
                _ => Ok(None)
            }

        }
        _ => Ok(None)
    };
}


fn get_rollup_function(fe: &FuncExpr) -> RuntimeResult<RollupFunction> {
    match RollupFunction::from_str(&fe.name) {
        Ok(rf) => Ok(rf),
        _ => {
            // should not happen
            Err(
                RuntimeError::General(
                    format!("Invalid rollup function \"{}\"", fe.name)
                )
            )
        }
    }
}

pub(super) fn get_timeseries_limit(aggr_expr: &AggrFuncExpr) -> RuntimeResult<usize> {
    // Incremental aggregates require holding only GOMAXPROCS timeseries in memory.
    let timeseries_len = usize::from(num_cpus()?);
    let res = if aggr_expr.limit > 0 {
        // There is an explicit limit on the number of output time series.
        timeseries_len * aggr_expr.limit
    } else {
        // Increase the number of timeseries for non-empty group list: `aggr() by (something)`,
        // since each group can have own set of time series in memory.
        timeseries_len * 1000
    };

    Ok(res)
}