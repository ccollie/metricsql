use std::sync::Arc;

use tracing::info;
use metricsql::common::ValueType;

use metricsql::functions::{Signature, TypeSignature, Volatility};
use metricsql::ast::{Expr};

use crate::context::Context;
use crate::eval::eval::eval_volatility;
use crate::eval::{create_evaluators, Evaluator, ExprEvaluator};
use crate::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

pub(crate) struct ArgList {
    args: Vec<ExprEvaluator>,
    parallel: bool,
    pub volatility: Volatility,
}

impl ArgList {
    pub fn new(signature: &Signature, args: &[Expr]) -> RuntimeResult<Self> {
        let _args = create_evaluators(args)?;
        let volatility = eval_volatility(signature, &_args);
        Ok(Self {
            args: _args,
            volatility,
            parallel: should_parallelize_param_parsing(signature),
        })
    }

    pub fn from(signature: &Signature, args: Vec<ExprEvaluator>) -> Self {
        let volatility = eval_volatility(signature, &args);
        Self {
            args,
            volatility,
            parallel: should_parallelize_param_parsing(signature),
        }
    }

    /// Are all arguments scalar/string or duration without step ?
    pub(super) fn all_const(&self) -> bool {
        self.args.len() == 0 || self.args.iter().all(|x| x.is_const())
    }

    pub fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<Vec<QueryValue>> {
        // todo: use tinyvec/heapless and pass in as &mut vec
        if self.parallel && self.args.len() >= 2 {
            return self.eval_parallel(ctx, ec);
        } else {
            let mut res: Vec<QueryValue> = Vec::with_capacity(self.args.len());
            for expr in self.args.iter() {
                res.push(expr.eval(ctx, ec)?);
            }
            Ok(res)
        }
    }

    fn eval_parallel(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<Vec<QueryValue>> {
        info!("eval function args in parallel");

        let params: _ = self
            .args
            .par_iter()
            .map(move |expr| expr.eval(&mut ctx.clone(), ec))
            .collect::<Vec<_>>();

        let mut result: Vec<QueryValue> = Vec::with_capacity(params.len());
        for p in params.into_iter() {
            match p {
                Ok(v) => result.push(v),
                Err(e) => return Err(e.clone()),
            }
        }

        Ok(result)
    }
}

#[inline]
fn should_parallelize(t: &ValueType) -> bool {
    !matches!(t, ValueType::String | ValueType::Scalar)
}

/// Determines if we should parallelize parameter evaluation. We ignore "lightweight"
/// parameter types like `String` or `Scalar`
fn should_parallelize_param_parsing(signature: &Signature) -> bool {
    let types = &signature.type_signature;

    fn check_args(valid_types: &[ValueType]) -> bool {
        valid_types
            .iter()
            .filter(|x| should_parallelize(*x))
            .count()
            > 1
    }

    return match types {
        TypeSignature::Variadic(valid_types, _min) => check_args(valid_types),
        TypeSignature::Uniform(number, valid_type) => {
            let types = &[valid_type.clone()];
            return *number >= 2 && check_args(types);
        }
        TypeSignature::VariadicEqual(data_type, _) => {
            let types = &[data_type.clone()];
            check_args(types)
        }
        TypeSignature::Exact(valid_types) => check_args(valid_types),
        TypeSignature::Any(number) => {
            if *number < 2 {
                return false;
            }
            true
        }
    };
}
