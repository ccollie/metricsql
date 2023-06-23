use metricsql::ast::FunctionExpr;
use metricsql::common::{Value, ValueType};
use metricsql::functions::{BuiltinFunction, Volatility};
use std::sync::Arc;
use tracing::{field, trace_span, Span};

use crate::context::Context;
use crate::eval::arg_list::ArgList;
use crate::eval::traits::Evaluator;
use crate::functions::transform::{get_transform_func, TransformFuncArg, TransformFuncHandler};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, QueryValue};

pub struct TransformEvaluator {
    fe: FunctionExpr,
    handler: TransformFuncHandler,
    args: ArgList,
    keep_metric_names: bool,
    return_type: ValueType,
    pub may_sort_results: bool,
}

impl TransformEvaluator {
    pub fn new(fe: &FunctionExpr) -> RuntimeResult<Self> {
        let function = match fe.function {
            BuiltinFunction::Transform(tf) => Ok(tf),
            _ => Err(RuntimeError::General(format!(
                "Error constructing TransformEvaluator: {} is not a transform fn",
                fe.name
            ))),
        }?;
        let handler = get_transform_func(function);
        let signature = function.signature();

        // todo: validate count
        let args = ArgList::new(&signature, &fe.args)?;
        let keep_metric_names = fe.keep_metric_names || function.keep_metric_name();

        let rv = fe.return_type();
        let return_type = ValueType::try_from(rv).unwrap_or(ValueType::RangeVector);

        Ok(Self {
            handler,
            args,
            fe: fe.clone(),
            keep_metric_names,
            return_type,
            may_sort_results: function.may_sort_results(),
        })
    }

    pub fn is_idempotent(&self) -> bool {
        self.volatility() != Volatility::Volatile && self.args.all_const()
    }
}

impl Value for TransformEvaluator {
    fn value_type(&self) -> ValueType {
        self.return_type.clone()
    }
}

impl Evaluator for TransformEvaluator {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
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
                Ok(QueryValue::InstantVector(v))
            }
        }
    }

    fn volatility(&self) -> Volatility {
        self.args.volatility
    }

    fn return_type(&self) -> ValueType {
        self.return_type.clone()
    }
}
