use crate::exec::remove_empty_series;
use crate::functions::arg_parse::{get_int_arg, get_series_arg};
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let limit: usize;
    let offset: usize;

    match tfa.args[0].get_int() {
        Err(e) => {
            return Err(RuntimeError::ArgumentError(format!(
                "cannot obtain limit arg: {e:?}"
            )));
        }
        Ok(l) => {
            limit = l as usize;
        }
    };

    match get_int_arg(&tfa.args, 1) {
        Err(_) => {
            return Err(RuntimeError::from("cannot obtain offset arg"));
        }
        Ok(v) => {
            offset = v as usize;
        }
    }

    let mut rvs = get_series_arg(&tfa.args, 2, tfa.ec)?;

    // remove_empty_series so offset will be calculated after empty series
    // were filtered out.
    remove_empty_series(&mut rvs);

    if rvs.len() >= offset {
        rvs.drain(0..offset);
    }
    if rvs.len() > limit {
        rvs.resize(limit, Timeseries::default());
    }

    Ok(std::mem::take(&mut rvs))
}
