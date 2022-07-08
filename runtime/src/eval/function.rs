use metricsql::ast::{FuncExpr};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::{create_evaluators, eval_args, ExprEvaluator};
use crate::eval::rollup::RollupEvaluator;
use crate::eval::traits::Evaluator;
use crate::functions::{
    rollup::{get_rollup_func},
    transform::{get_transform_func, TransformFunc, TransformFuncArg}
};

use crate::timeseries::Timeseries;
use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(super) fn create_function_evaluator(fe: &FuncExpr) -> RuntimeResult<ExprEvaluator> {
    if let Some(func) = get_rollup_func(&fe.name) {
        Ok(ExprEvaluator::Rollup(RollupEvaluator::from_function(fe)?))
    } else {
        let fe = TransformEvaluator::new(fe)?;
        Ok(ExprEvaluator::Function(fe))
    }
}

pub(super) struct TransformEvaluator {
    fe: FuncExpr,
    transform_func: &'static TransformFunc,
    args: Vec<ExprEvaluator>
}

impl TransformEvaluator {
    pub fn new(fe: &FuncExpr) -> RuntimeResult<Self> {
        match get_transform_func(&fe.name) {
            None => panic!("Bug: unknown func {}", fe.name),
            Some(transform_func) => {
                let args = create_evaluators(&fe.args)?;
                Ok(Self {
                    transform_func,
                    args,
                    fe: fe.clone()
                })
            }
        }
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &mut EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo: tinyvec
        let args = eval_args(ctx, ec, &self.args)?;
        let mut tfa = TransformFuncArg::new(ec, self.fe, &args);
        match (self.transform_func)(&mut tfa) {
            Err(err) => Err(RuntimeError::from(format!("cannot evaluate {}: {}", self.fe, err))),
            Ok(v) => Ok(v)
        }
    }
}
