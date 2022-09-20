use metricsql::ast::FuncExpr;
use metricsql::functions::{BuiltinFunction, Signature, TransformFunction, Volatility};

use crate::context::Context;
use crate::eval::{create_evaluators, eval_params, ExprEvaluator};
use crate::eval::eval::eval_volatility;
use crate::eval::rollup::RollupEvaluator;
use crate::eval::traits::Evaluator;
use crate::EvalConfig;
use crate::functions::{
    transform::{get_transform_func, TransformFn, TransformFuncArg}
};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::Timeseries;

pub(super) fn create_function_evaluator(fe: &FuncExpr) -> RuntimeResult<ExprEvaluator> {
    if fe.is_rollup() {
        Ok(ExprEvaluator::Rollup(RollupEvaluator::from_function(fe)?))
    } else {
        let fe = TransformEvaluator::new(fe)?;
        Ok(ExprEvaluator::Function(fe))
    }
}

pub(super) struct TransformEvaluator {
    fe: FuncExpr,
    function: TransformFunction,
    /// function signature. Copied here to avoid per-call allocation
    signature: Signature,
    handler: Box<dyn TransformFn + 'static>,
    args: Vec<ExprEvaluator>,
    keep_metric_names: bool
}

impl TransformEvaluator {
    pub fn new(fe: &FuncExpr) -> RuntimeResult<Self> {
        match fe.function {
            BuiltinFunction::Transform(function) => {
                let func = get_transform_func(function)?;
                let signature = function.signature();

                // todo: validate count
                let args = create_evaluators(&fe.args)?;
                let keep_metric_names = fe.keep_metric_names || function.keep_metric_name();
                Ok(Self {
                    handler: Box::new(func),   // looks sketchy
                    args,
                    function,
                    signature,
                    fe: fe.clone(),
                    keep_metric_names
                })
            },
            _ => {
                // todo: use a specific variant
                return Err(RuntimeError::General(
                    format!("Error constructing TransformEvaluator: {} is not a transform fn", fe.name())
                ))
            }
        }
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        // todo: tinyvec
        let args = eval_params(ctx, ec, &self.signature.type_signature, &self.args)?;
        let mut tfa = TransformFuncArg::new(ec, &self.fe, &args, self.keep_metric_names);
        match (self.handler)(&mut tfa) {
            Err(err) => Err(RuntimeError::from(format!("cannot evaluate {}: {}", self.fe, err))),
            Ok(v) => Ok(v)
        }
    }

    fn volatility(&self) -> Volatility {
        eval_volatility(&self.signature, &self.args)
    }
}
