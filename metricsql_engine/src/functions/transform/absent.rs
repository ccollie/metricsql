use crate::execution::{eval_number, EvalConfig};
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{InstantVector, RuntimeResult, Timeseries};

pub(crate) fn transform_absent(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    let rvs = handle_absent(&series, tfa.ec)?;
    // set_labels_from_arg(&mut rvs, &tfa.fe.args[0]);
    Ok(rvs)
}

pub(crate) fn handle_absent(
    series: &InstantVector,
    ec: &EvalConfig,
) -> RuntimeResult<InstantVector> {
    let mut rvs = eval_number(ec, 1.0)?;
    if series.is_empty() {
        return Ok(rvs);
    }

    for i in 0..series[0].values.len() {
        let is_absent = series.iter().all(|ts| ts.values[i].is_nan());
        if !is_absent {
            rvs[0].values[i] = f64::NAN
        }
    }

    Ok(rvs)
}
