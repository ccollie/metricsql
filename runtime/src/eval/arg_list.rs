use std::sync::Arc;
use rayon::iter::{IntoParallelRefIterator};

use metricsql::ast::BExpression;
use metricsql::functions::{DataType, Signature, TypeSignature, Volatility};

use crate::{EvalConfig};
use crate::context::Context;
use crate::eval::{create_evaluators, Evaluator, ExprEvaluator};
use crate::eval::eval::eval_volatility;
use crate::functions::types::AnyValue;
use crate::rayon::iter::ParallelIterator;
use crate::runtime_error::{RuntimeResult};

pub(crate) struct ArgList {
    args: Vec<ExprEvaluator>,
    parallel: bool,
    pub volatility: Volatility,
}

impl ArgList {
    pub fn new(signature: &Signature, args: &[BExpression]) -> RuntimeResult<Self> {
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

    pub fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<Vec<AnyValue>> {
        // todo: use tinyvec and pass in as &mut vec
        if self.parallel && self.args.len() >= 2 {
            return self.eval_parallel(ctx, ec)
        } else {
            let mut res: Vec<AnyValue> = Vec::with_capacity(self.args.len());
            for expr in self.args.iter() {
                res.push(expr.eval(ctx, ec)? );
            }
            Ok(res)
        }
    }

    fn eval_parallel(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<Vec<AnyValue>> {
        let params: _ = self.args.par_iter()
            .map(move |expr| { expr.eval(&mut ctx.clone(), ec) })
            .collect::<Vec<_>>();

        let mut result: Vec<AnyValue> = Vec::with_capacity(params.len());
        for p in params.into_iter() {
            match p {
                Ok(v) => {
                    result.push(v)
                },
                Err(e) => {
                    return Err(e.clone())
                }
            }
        }

        Ok(result)
    }
}


#[inline]
fn should_parallelize(t: &DataType) -> bool {
    !matches!(t, DataType::String | DataType::Scalar )
}

/// Determines if we should parallelize parameter evaluation. We ignore "lightweight"
/// parameter types like `String` or `Int`
pub(crate) fn should_parallelize_param_parsing(signature: &Signature) -> bool {
    let types = &signature.type_signature;

    fn check_args(valid_types: &[DataType]) -> bool {
        valid_types.iter().filter(|x| should_parallelize(*x)).count() > 1
    }

    return match types {
        TypeSignature::Variadic(valid_types, _min) => {
            check_args(valid_types)
        },
        TypeSignature::Uniform(number, valid_type) => {
            let types = &[*valid_type];
            return *number >= 2 && check_args(types)
        },
        TypeSignature::VariadicEqual(data_type, _) => {
            let types = &[*data_type];
            check_args(types)
        }
        TypeSignature::Exact(valid_types) => {
            check_args(valid_types)
        },
        TypeSignature::Any(number) => {
            if *number < 2 {
                return false
            }
            true
        }
    }
}
