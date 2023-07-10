use metricsql::common::ValueType;
use metricsql::functions::{Signature, TypeSignature};

use crate::{QueryValue, Timeseries};

pub(crate) fn series_len(val: &QueryValue) -> usize {
    match &val {
        QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => iv.len(),
        _ => 1,
    }
}

#[inline]
pub fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| !ts.values.iter().all(|v| v.is_nan()));
}

#[inline]
fn should_parallelize(t: &ValueType) -> bool {
    !matches!(t, ValueType::String | ValueType::Scalar)
}

/// Determines if we should parallelize parameter evaluation. We ignore "lightweight"
/// parameter types like `String` or `Scalar`
fn should_parallelize_param_eval(signature: &Signature) -> bool {
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
            let types = &[*data_type];
            check_args(types)
        }
        TypeSignature::Exact(valid_types) => check_args(valid_types),
        TypeSignature::Any(number) => {
            if *number < 2 {
                return false;
            }
            true
        }
        TypeSignature::VariadicAny(x) => {
            // todo: better heuristic
            if *x < 2 {
                return false;
            }
            true
        }
    };
}
