use std::borrow::Cow;

use crate::functions::utils::float_to_int_bounded;
use crate::{EvalConfig, QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn get_string_arg(args: &[QueryValue], arg_num: usize) -> RuntimeResult<Cow<String>> {
    if arg_num > args.len() - 1 {
        let msg = format!("missing string arg # {}", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    let res = match &args[arg_num] {
        QueryValue::String(s) => Ok(Cow::Borrowed(s)), // todo: use .into ??
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
            // todo: return reference
            let res = Cow::Owned(series[0].metric_name.metric_group.clone());
            return Ok(res);
        }
        _ => Err(RuntimeError::ArgumentError(
            "string parameter expected ".to_string(),
        )),
    };
    res
}

pub(crate) fn get_series_arg(
    args: &[QueryValue],
    arg_num: usize,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<Timeseries>> {
    if let Some(arg) = args.get(arg_num) {
        return arg.get_instant_vector(ec);
    }
    let msg = format!("missing series arg # {}", arg_num + 1);
    return Err(RuntimeError::ArgumentError(msg));
}

// TODO: COW, or return Iterator
pub(crate) fn get_scalar_arg_as_vec(
    args: &[QueryValue],
    arg_num: usize,
    ec: &EvalConfig,
) -> RuntimeResult<Vec<f64>> {
    // todo: check bounds
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
            return Err(RuntimeError::ArgumentError(msg));
        }
    }
}

pub(crate) fn get_scalar_arg_iter<'a>(
    args: &[QueryValue],
    arg_num: usize,
    ec: &EvalConfig,
) -> RuntimeResult<impl Iterator<Item = f64>> {
    // todo: check bounds
    let arg = args.get(arg_num);
    if arg.is_none() {
        let msg = format!("missing scalar arg # {}", arg_num + 1);
        return Err(RuntimeError::ArgumentError(msg));
    }
    let arg = arg.unwrap();
    match arg {
        QueryValue::Scalar(val) => {
            let len = ec.data_points();
            let iter = std::iter::repeat(*val).take(len);
            Ok(iter)
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
            Ok(s[0].values.iter())
        }
        _ => {
            let msg = format!(
                "arg # {} expected float or a single timeseries; got {}",
                arg_num + 1,
                arg.data_type()
            );
            return Err(RuntimeError::ArgumentError(msg));
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
        QueryValue::Scalar(val) => {
            let v = *val;
            return Ok(v);
        }
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
    return Err(RuntimeError::ArgumentError(msg));
}

pub(crate) fn get_int_arg(args: &[QueryValue], arg_num: usize) -> RuntimeResult<i64> {
    get_float_arg(args, arg_num, Some(0_f64)).map(float_to_int_bounded)
}
