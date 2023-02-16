use metricsql::functions::{DataType, Volatility};
use metricsql::hir::FunctionExpr;
use metricsql::prelude::TransformFunction;
use std::str::FromStr;
use std::sync::Arc;
use tracing::{field, trace_span, Span};

use crate::context::Context;
use crate::eval::arg_list::ArgList;
use crate::eval::rollup::RollupEvaluator;
use crate::eval::traits::Evaluator;
use crate::eval::ExprEvaluator;
use crate::functions::transform::TransformFnImplementation;
use crate::functions::transform::{get_transform_func, TransformFuncArg};
use crate::functions::types::AnyValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::EvalConfig;

pub(super) fn create_function_evaluator(fe: &FunctionExpr) -> RuntimeResult<ExprEvaluator> {
    let fe = TransformEvaluator::new(fe)?;
    Ok(ExprEvaluator::Function(fe))
}

pub struct TransformEvaluator {
    fe: FunctionExpr,
    handler: TransformFnImplementation,
    args: ArgList,
    keep_metric_names: bool,
    return_type: DataType,
}

impl TransformEvaluator {
    pub fn new(fe: &FunctionExpr) -> RuntimeResult<Self> {
        match TransformFunction::from_str(&fe.name) {
            Ok(function) => {
                let handler = get_transform_func(function)?;
                let signature = function.signature();

                // todo: validate count
                let args = ArgList::new(&signature, &fe.args)?;
                let keep_metric_names = fe.keep_metric_names || function.keep_metric_name();

                let rv = fe.return_type();
                let return_type = DataType::try_from(rv).unwrap_or(DataType::RangeVector);

                Ok(Self {
                    handler,
                    args,
                    fe: fe.clone(),
                    keep_metric_names,
                    return_type,
                })
            }
            _ => {
                // todo: use a specific variant
                return Err(RuntimeError::General(format!(
                    "Error constructing TransformEvaluator: {} is not a transform fn",
                    fe.name
                )));
            }
        }
    }

    pub fn is_idempotent(&self) -> bool {
        self.volatility() != Volatility::Volatile && self.args.all_const()
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        let span = if ctx.trace_enabled() {
            trace_span!("transform", function = self.fe.name, series = field::Empty)
        } else {
            Span::none()
        }
        .entered();

        let args = self.args.eval(ctx, ec)?;
        let mut tfa = TransformFuncArg::new(ec, &self.fe, args, self.keep_metric_names);

        match (self.handler)(&mut tfa) {
            Err(err) => {
                span.record("series", 0);
                Err(RuntimeError::from(format!(
                    "cannot evaluate {}: {:?}",
                    self.fe, err
                )))
            }
            Ok(v) => {
                span.record("series", v.len());
                Ok(AnyValue::InstantVector(v))
            }
        }
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }

    fn return_type(&self) -> DataType {
        self.return_type
    }
}
