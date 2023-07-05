use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{QueryValue, Timeseries};

#[inline]
pub(crate) fn get_single_timeseries(series: &Vec<Timeseries>) -> RuntimeResult<&Timeseries> {
    if series.len() != 1 {
        let msg = format!(
            "arg must contain a single timeseries; got {} timeseries",
            series.len()
        );
        return Err(RuntimeError::TypeCastError(msg));
    }
    Ok(&series[0])
}

pub fn get_scalar_param_value(
    param: &QueryValue,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<f64> {
    match param {
        QueryValue::Scalar(val) => Ok(*val),
        _ => {
            let msg = format!(
                "expected scalar arg for parameter \"{}\" of function {}; Got {}",
                param_name,
                func_name,
                param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg));
        }
    }
}

#[inline]
pub fn get_string_param_value(
    param: &QueryValue,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<String> {
    match param {
        QueryValue::String(val) => Ok(val.clone()),
        _ => {
            let msg = format!(
                "expected string arg for parameter \"{}\" of function {}; Got {}",
                param_name,
                func_name,
                param.data_type_name()
            );
            return Err(RuntimeError::TypeCastError(msg));
        }
    }
}
