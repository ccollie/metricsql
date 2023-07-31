use metricsql::parser::parse_number;

use crate::eval::eval_number;
use crate::functions::transform::TransformFuncArg;
use crate::{QueryValue, RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn scalar(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let arg = get_arg(tfa, 0)?;
    match arg {
        // Verify whether the arg is a string.
        // Then try converting the string to number.
        QueryValue::String(s) => {
            let n = parse_number(s).map_or_else(|_| f64::NAN, |n| n);
            eval_number(tfa.ec, n)
        }
        QueryValue::Scalar(f) => eval_number(tfa.ec, *f),
        _ => {
            // The arg isn't a string. Extract scalar from it.
            if tfa.args.len() != 1 {
                eval_number(tfa.ec, f64::NAN)
            } else {
                let mut arg = arg.get_instant_vector(tfa.ec)?.remove(0);
                arg.metric_name.reset();
                Ok(vec![arg])
            }
        }
    }
}

fn get_arg<'a>(tfa: &'a TransformFuncArg, index: usize) -> RuntimeResult<&'a QueryValue> {
    if index >= tfa.args.len() {
        return Err(RuntimeError::ArgumentError(format!(
            "expected at least {} args; got {}",
            index + 1,
            tfa.args.len()
        )));
    }
    Ok(&tfa.args[index])
}
