use std::str::FromStr;

use rayon::iter::IntoParallelRefIterator;

use metricsql::functions::{DataType, Signature, TypeSignature};

use crate::{EvalConfig, Timeseries};
use crate::context::Context;
use crate::eval::ExprEvaluator;
use crate::functions::types::ParameterValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};

pub(crate) struct ArgList {
    args: Vec<ExprEvaluator>,
    parallel: bool
}

impl ArgList {
    pub fn new(signature: &Signature, args: Vec<ExprEvaluator>) -> Self {
        Self {
            args,
            parallel: should_parallelize_param_parsing(signature)
        }
    }

    pub fn eval(&mut self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<ParameterValue>> {
        if self.parallel {
            todo!()
        } else {
            todo!()
        }
    }

    fn eval_parallel(&mut self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<ParameterValue>> {
        // self.args.par_iter().map(|x| {
        //     eval_param(ctx, ec)
        // })
        todo!()
    }
}

struct IndexedItem<T> {
    item: T,
    index: usize
}


#[inline]
fn should_parallelize(t: &DataType) -> bool {
    !matches!(t, DataType::String | DataType::Float | DataType::Int | DataType::Vector )
}

/// Determines if we should parallelize parameter parsing. We ignore "lightweight"
/// parameter types like `String` or `Int`
pub(crate) fn should_parallelize_param_parsing(signature: &Signature) -> bool {
    let types = &signature.type_signature;

    fn check_args(valid_types: &[DataType]) -> bool {
        valid_types.iter().filter(|x| should_parallelize(*x)).count() > 1
    }

    return match types {
        TypeSignature::Variadic(valid_types, min) => {
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


// todo: use COW for this ??
fn convert(src: &ParameterValue, expected: DataType) -> RuntimeResult<ParameterValue> {
    match expected {
        DataType::Matrix => convert_to_matrix(src),
        DataType::Series => convert_to_series(src),
        DataType::Vector => convert_to_vector(src),
        DataType::Float => convert_to_float(src),
        DataType::Int => convert_to_int(src),
        DataType::String => convert_to_string(src)
    }
}

fn convert_to_vector(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    todo!()
}

fn convert_to_matrix(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    todo!()
}

fn convert_to_series(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    todo!()
}

fn convert_to_string(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    match src {
        ParameterValue::String(s) => {
            Ok(ParameterValue::String(s.to_string()))
        }
        ParameterValue::Series(series) => {
            let ts = get_single_timeseries(series)?;
            if ts.values.len() > 0 {
                let all_nan = series[0].values.iter().all(|x| x.is_nan());
                if !all_nan {
                    let msg = format!("series contains non-string timeseries");
                    return Err(RuntimeError::ArgumentError(msg));
                }
            }
            let res = ts.metric_name.metric_group.clone();
            // todo: return reference
            Ok(ParameterValue::String(res))
        }
        _ => {
            let msg = format!("cannot cast {} to a string", src.data_type());
            return Err(RuntimeError::TypeCastError(msg))
        }
    }
}

fn convert_to_float(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    match src {
        ParameterValue::Series(series) => {
            let ts = get_single_timeseries(series)?;
            if ts.values.len() != 1 {
                let msg = format!("expected a vector of size 1; got {}", ts.values.len());
                return Err(RuntimeError::ArgumentError(msg))
            }
            Ok(ParameterValue::Float(ts.values[0]))
        },
        ParameterValue::Vector(vec) => {
            if vec.len() != 1 {
                let msg = format!("expected a vector of size 1; got {}", vec.len());
                return Err(RuntimeError::ArgumentError(msg))
            }
            Ok(ParameterValue::Float(vec[0]))
        },
        ParameterValue::Float(val) => Ok(ParameterValue::Float(*val)),
        ParameterValue::Int(val) => Ok(ParameterValue::Float(*val as f64)),
        ParameterValue::String(s) => {
            match f64::from_str(s) {
                Err(e) => {
                    return Err(RuntimeError::TypeCastError(
                        format!("{} cannot be converted to a float", s)
                    ))
                },
                Ok(val) => Ok( ParameterValue::Float(val))
            }
        },
        _ => {
            return Err(RuntimeError::TypeCastError(
                format!("{} cannot be converted to a float", src.data_type())
            ))
        }
    }
}

fn convert_to_int(src: &ParameterValue) -> RuntimeResult<ParameterValue> {
    match src {
        ParameterValue::Int(val) => Ok(ParameterValue::Int(*val)),
        ParameterValue::Float(val) => Ok(ParameterValue::Int(*val as i64)),
        _=> {
            match convert_to_float(src)? {
                ParameterValue::Float(f) => Ok(ParameterValue::Int(f as i64)),
                _=> unreachable!()
            }
        }
    }
}


#[inline]
fn get_single_series_from_param(param: &ParameterValue) -> RuntimeResult<&Timeseries> {
    match param {
        ParameterValue::Series(series) => get_single_timeseries(series),
        _ => {
            let msg = format!("expected a timeseries vector; got {}", param.data_type());
            return Err(RuntimeError::TypeCastError(msg))
        }
    }
}

#[inline]
fn get_single_timeseries(series: &Vec<Timeseries>) -> RuntimeResult<&Timeseries> {
    if series.len() != 1 {
        let msg = format!(
            "arg must contain a single timeseries; got {} timeseries",
            series.len()
        );
        return Err(RuntimeError::TypeCastError(msg))
    }
    Ok(&series[0])
}
