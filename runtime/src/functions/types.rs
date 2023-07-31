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
    args: &[QueryValue],
    index: usize,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<f64> {
    if let Some(QueryValue::Scalar(val)) = args.get(index) {
        return Ok(*val);
    }
    let msg = format!("expected scalar arg for parameter '{param_name}' of function {func_name};",);
    Err(RuntimeError::TypeCastError(msg))
}

#[inline]
pub fn get_string_param_value(
    args: &[QueryValue],
    arg_num: usize,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<String> {
    let param = args.get(arg_num);
    if param.is_none() {
        let msg = format!(
            "expected string arg for parameter \"{}\" of function {}; Got None",
            param_name, func_name
        );
        return Err(RuntimeError::TypeCastError(msg));
    }
    let param = param.unwrap();
    match param {
        QueryValue::String(val) => Ok(val.clone()),
        _ => {
            let msg = format!(
                "expected string arg for parameter \"{param_name}\" of function {func_name}; Got {}",
                param
            );
            Err(RuntimeError::TypeCastError(msg))
        }
    }
}
