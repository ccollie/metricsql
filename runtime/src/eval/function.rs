use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use metricsql::types::*;

use crate::{EvalConfig, get_timeseries, MetricName};
use crate::aggr::{AggrFuncArg, get_aggr_func};
use crate::eval::{create_evaluator};
use crate::eval::rollup::RollupEvaluator;
use crate::eval::traits::Evaluator;
use crate::rollup::{
    get_rollup_configs,
    get_rollup_func,
    NewRollupFunc,
    RollupFunc
};
use crate::timeseries::Timeseries;
use crate::transform::{get_absent_timeseries, get_transform_func, TransformFunc, TransformFuncArg};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::get_rollup_func;

pub(super) fn create_function_evaluator(fe: &FunctionExpr) -> Box<dyn Evaluator> {
    if Some(func) = get_rollup_func(&fe.name) {
        Box::new(RollupEvaluator::from_function(fe))
    } else {
        Box::new(TransformEvaluator::new(fe))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct TransformEvaluator {
    fe: FuncExpr,
    transform_func: &'static TransformFunc,
    args: Vec<dyn Evaluator>
}

impl TransformEvaluator {
    pub fn new(fe: &FuncExpr) -> Self {
        match get_transform_func(&fe.name) {
            None => panic!("Bug: unknown func {}", fe.name),
            Some(transform_func) => {
                let args = fe.args.map(|x| create_evaluator(x)).collect();
                Self {
                    transform_func,
                    args,
                    fe: fe.clone()
                }
            }
        }
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ec: &mut EvalConfig) -> RuntimeResult<&Vec<Timeseries>> {
        let args = self.args.map(|x| x.eval(ec)?).collect();
        let mut tfa = TransformFuncArg::new(ec, self.fe, args);
        match self.transform_func(&mut tfa) {
            Err(err) => Err(RuntimeError::from(format!("cannot evaluate {}: {}", self.fe, err))),
            Ok(v) => Ok(v)
        }
    }
}
