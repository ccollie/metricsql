use metricsql::ast::Expr;

use crate::eval::eval_number;
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{EvalConfig, RuntimeResult, Timeseries};

pub(crate) fn absent(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut rvs = get_absent_timeseries(tfa.ec, &tfa.fe.args[0])?;

    let series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    if series.is_empty() {
        return Ok(rvs);
    }

    for i in 0..series[0].values.len() {
        let mut is_absent = true;
        for ts in series.iter() {
            if !ts.values[i].is_nan() {
                is_absent = false;
                break;
            }
        }
        if !is_absent {
            rvs[0].values[i] = f64::NAN
        }
    }
    Ok(rvs)
}

pub(crate) fn get_absent_timeseries(ec: &EvalConfig, arg: &Expr) -> RuntimeResult<Vec<Timeseries>> {
    // Copy tags from arg
    let mut rvs = eval_number(ec, 1.0)?;
    if let Expr::MetricExpression(me) = arg {
        for tf in me.label_filters.iter() {
            if tf.label.is_empty() {
                continue;
            }
            if tf.is_regexp() || tf.is_negative() {
                continue;
            }
            rvs[0].metric_name.set_tag(&tf.label, &tf.value)
        }
    }
    Ok(rvs)
}
