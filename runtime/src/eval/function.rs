use std::sync::Arc;
use metricsql::ast::FuncExpr;
use metricsql::functions::{BuiltinFunction, DataType, Volatility};

use crate::context::Context;
use crate::eval::ExprEvaluator;
use crate::eval::arg_list::ArgList;
use crate::eval::rollup::RollupEvaluator;
use crate::eval::traits::Evaluator;
use crate::EvalConfig;
use crate::functions::{
    transform::{
        get_transform_func,
        TransformFuncArg
    }
};
use crate::functions::transform::TransformFnImplementation;
use crate::functions::types::AnyValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(super) fn create_function_evaluator(fe: &FuncExpr) -> RuntimeResult<ExprEvaluator> {
    if fe.is_rollup() {
        Ok(ExprEvaluator::Rollup(RollupEvaluator::from_function(fe)?))
    } else {
        let fe = TransformEvaluator::new(fe)?;
        Ok(ExprEvaluator::Function(fe))
    }
}

pub struct TransformEvaluator {
    fe: FuncExpr,
    handler: TransformFnImplementation,
    args: ArgList,
    keep_metric_names: bool,
    return_type: DataType,
}

impl TransformEvaluator {
    pub fn new(fe: &FuncExpr) -> RuntimeResult<Self> {
        match fe.function {
            BuiltinFunction::Transform(function) => {
                let handler = get_transform_func(function)?;
                let signature = function.signature();

                // todo: validate count
                let args = ArgList::new(&signature, &fe.args)?;
                let keep_metric_names = fe.keep_metric_names || function.keep_metric_name();

                let rv = fe.return_value();
                let return_type = DataType::try_from(rv).unwrap_or(DataType::RangeVector);

                Ok(Self {
                    handler,
                    args,
                    fe: fe.clone(),
                    keep_metric_names,
                    return_type
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

    pub fn is_idempotent(&self) -> bool {
        self.volatility() != Volatility::Volatile && self.args.all_const()
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        let args = self.args.eval(ctx, ec)?;
        let mut tfa = TransformFuncArg::new(ec, &self.fe, args, self.keep_metric_names);
        match (self.handler)(&mut tfa) {
            Err(err) => Err(RuntimeError::from(format!("cannot evaluate {}: {:?}", self.fe, err))),
            Ok(v) => Ok(AnyValue::InstantVector(v))
        }
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }

    fn return_type(&self) -> DataType {
        self.return_type
    }
}
