use crate::functions::transform::{transform_series, TransformFuncArg};
use crate::{RuntimeError, RuntimeResult};
use crate::types::{QueryValue, Timeseries};

pub(crate) fn round(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let args_len = tfa.args.len();

    if !(1..=2).contains(&args_len) {
        return Err(RuntimeError::ArgumentError(format!(
            "unexpected number of arguments: #{}; want 1 or 2",
            tfa.args.len()
        )));
    }

    if args_len == 1 {
        return transform_series(tfa, |values: &mut [f64]| {
            prometheus_round(values, &[1.0]);
        });
    }

    if let Some(nearest_arg) = tfa.args.get(1) {
        match nearest_arg {
            QueryValue::Scalar(val) => {
                let v = *val;
                return transform_series(tfa, |values: &mut [f64]| {
                    prometheus_round(values, &[v]);
                });
            }
            QueryValue::InstantVector(s) => {
                if s.len() != 1 {
                    let msg = format!(
                        "arg #2 must contain a single timeseries; got {} timeseries",
                        s.len()
                    );
                    return Err(RuntimeError::ArgumentError(msg));
                }
                let vals = s[0].values.clone();
                return transform_series(tfa, move |values: &mut [f64]| {
                    prometheus_round(values, &vals);
                });
            }
            _ => {}
        }
    }
    let err_msg = "Scalar expected as second argument to round function";
    Err(RuntimeError::ArgumentError(err_msg.to_string()))
}

fn prometheus_round(vals: &mut [f64], nearest: &[f64]) {
    // round returns a number rounded to to_nearest.
    // Ties are solved by rounding up.
    if nearest.len() == 1 {
        let to_nearest = nearest[0];
        // Invert as it seems to cause fewer floating point accuracy issues.
        let to_nearest_inverse = 1.0 / to_nearest;

        for v in vals.iter_mut() {
            let f = (*v * to_nearest_inverse + 0.5).floor() / to_nearest_inverse;
            *v = f;
        }
    } else {
        for (v, n) in vals.iter_mut().zip(nearest) {
            // Invert as it seems to cause fewer floating point accuracy issues.
            let to_nearest_inverse = 1.0 / n;
            let f = (*v * to_nearest_inverse + 0.5).floor() / to_nearest_inverse;
            *v = f;
        }
    }
}
