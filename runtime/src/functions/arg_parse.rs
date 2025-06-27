use std::borrow::Cow;
use crate::types::QueryValue;
use crate::{RuntimeError, RuntimeResult};
use crate::execution::EvalConfig;
use crate::functions::utils::float_to_int_bounded;

pub(crate) fn get_string_arg(args: &[QueryValue], arg_num: usize) -> RuntimeResult<Cow<str>> {
    if arg_num > args.len() - 1 {
        let msg = format!("missing string arg # {}", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    match &args[arg_num] {
        QueryValue::String(s) => Ok(Cow::Borrowed(s)),
        QueryValue::Scalar(f) => Ok(Cow::Owned(f.to_string())),
        QueryValue::InstantVector(series) => {
            if series.len() != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    series.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            for v in series[0].values.iter() {
                if !v.is_nan() {
                    let msg = format!("arg # {} contains non - string timeseries", arg_num + 1);
                    return Err(RuntimeError::ArgumentError(msg));
                }
            }
            // Use the String directly as a Cow<str> instead of creating a Cow<String>
            let res = Cow::Owned(series[0].metric_name.measurement.clone());
            Ok(res)// This now returns Cow<str> as expected
        }
        _ => Err(RuntimeError::ArgumentError(format!(
            "string expected for parameter {arg_num}",
        ))),
    }
}

// TODO: COW, or return Iterator
pub(crate) fn get_scalar_arg_as_vec(
    args: &[QueryValue],
    arg_num: usize,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<f64>> {
    let arg = args.get(arg_num);
    if arg.is_none() {
        let msg = format!("missing scalar arg # {}", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    let arg = arg.unwrap();
    match arg {
        QueryValue::Scalar(val) => {
            // todo: use object pool here
            let len = ec.data_points();
            // todo: tinyvec
            let values = vec![*val; len];
            Ok(values)
        }
        QueryValue::InstantVector(s) => {
            if s.len() != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    s.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            Ok(s[0].values.clone())
        }
        _ => {
            let msg = format!(
                "arg # {} expected float or a single timeseries; got {}",
                arg_num + 1,
                arg.data_type()
            );
            Err(RuntimeError::ArgumentError(msg))
        }
    }
}

pub(crate) fn get_float_arg(
    args: &[QueryValue],
    arg_num: usize,
    default_value: Option<f64>,
) -> RuntimeResult<f64> {
    // todo: check bounds
    let arg = &args[arg_num];
    match arg {
        QueryValue::Scalar(val) => return Ok(*val),
        QueryValue::InstantVector(s) => {
            let len = s.len();
            if len == 0 {
                if let Some(default) = default_value {
                    return Ok(default);
                }
            }
            if len != 1 {
                let msg = format!(
                    "arg # {} must contain a single timeseries; got {} timeseries",
                    arg_num + 1,
                    s.len()
                );
                return Err(RuntimeError::ArgumentError(msg));
            }
            return Ok(s[0].values[0]);
        }
        _ => {}
    }

    let msg = format!(
        "arg # {} expected float or a single timeseries; got {}",
        arg_num + 1,
        arg.data_type()
    );
    Err(RuntimeError::ArgumentError(msg))
}

pub(crate) fn get_int_arg(args: &[QueryValue], arg_num: usize) -> RuntimeResult<i64> {
    get_float_arg(args, arg_num, Some(0_f64)).map(float_to_int_bounded)
}

#[inline]
pub(crate) fn get_string_param_value(
    args: &[QueryValue],
    arg_num: usize,
    func_name: &str,
    param_name: &str,
) -> RuntimeResult<String> {
    let param = args.get(arg_num);
    if param.is_none() {
        let msg = format!(
            "expected string arg for parameter \"{param_name}\" of function {func_name}; Got None",
        );
        return Err(RuntimeError::TypeCastError(msg));
    }
    let param = param.unwrap();
    match param {
        QueryValue::String(val) => Ok(val.clone()),
        _ => {
            let msg = format!(
                "expected string arg for parameter \"{param_name}\" of function {func_name}; Got {param}",
            );
            Err(RuntimeError::TypeCastError(msg))
        }
    }
}

pub(crate) fn get_scalar_param_value(
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
