use std::cmp::Ordering;

use crate::common::strings::compare_str_alphanumeric;
use crate::functions::arg_parse::get_series_arg;
use crate::functions::transform::TransformFuncArg;
use crate::{RuntimeError, RuntimeResult, Timeseries};

pub(crate) fn sort(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, false)
}

pub(crate) fn sort_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_sort_impl(tfa, true)
}

pub(crate) fn sort_alpha_numeric(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, false)
}

pub(crate) fn sort_alpha_numeric_desc(
    tfa: &mut TransformFuncArg,
) -> RuntimeResult<Vec<Timeseries>> {
    label_alpha_numeric_sort_impl(tfa, true)
}

pub(crate) fn sort_by_label(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, false)
}

pub(crate) fn sort_by_label_desc(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    sort_by_label_impl(tfa, true)
}

fn transform_sort_impl(
    tfa: &mut TransformFuncArg,
    is_desc: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    series.sort_by(move |first, second| {
        let a = &first.values;
        let b = &second.values;
        let iter_a = a.iter().rev();
        let iter_b = b.iter().rev();

        for (x, y) in iter_a.zip(iter_b) {
            if !x.is_nan() && !y.is_nan() && x != y {
                return if is_desc {
                    y.total_cmp(x)
                } else {
                    x.total_cmp(y)
                };
            }
        }
        Ordering::Equal
    });

    Ok(std::mem::take(&mut series))
}

fn sort_by_label_impl(tfa: &mut TransformFuncArg, is_desc: bool) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = Vec::with_capacity(1);
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    for arg in tfa.args.iter().skip(1) {
        let label = arg.get_string()?;
        labels.push(label);
    }

    series.sort_by(|first, second| {
        for label in labels.iter() {
            let a = first.metric_name.get_tag_value(&label);
            let b = second.metric_name.get_tag_value(&label);
            match (a, b) {
                (None, None) => continue,
                (Some(a1), Some(b1)) => {
                    if a1 == b1 {
                        continue;
                    } else if is_desc {
                        return b1.cmp(&a1);
                    } else {
                        return a1.cmp(&b1);
                    }
                }
                (Some(_), None) => return Ordering::Greater,
                (None, Some(_)) => return Ordering::Less,
            }
        }
        return Ordering::Equal;
    });

    Ok(series)
}

fn label_alpha_numeric_sort_impl(
    tfa: &mut TransformFuncArg,
    is_desc: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels: Vec<String> = vec![];
    for (i, arg) in tfa.args.iter().skip(1).enumerate() {
        let label = arg.get_string().map_err(|err| {
            RuntimeError::ArgumentError(format!(
                "cannot parse label {} for sorting: {:?}",
                i + 1,
                err
            ))
        })?;
        labels.push(label);
    }

    let mut res = get_series_arg(&tfa.args, 0, tfa.ec)?;
    res.sort_by(|first, second| {
        for label in &labels {
            match (
                first.metric_name.get_tag_value(&label),
                second.metric_name.get_tag_value(&label),
            ) {
                (None, None) => continue,
                (Some(a), Some(b)) => {
                    return if is_desc {
                        compare_str_alphanumeric(b, a)
                    } else {
                        compare_str_alphanumeric(a, b)
                    };
                }
                (None, Some(_)) => {
                    return if is_desc {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                }
                (Some(_), None) => {
                    return if is_desc {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
            }
        }
        Ordering::Equal
    });

    Ok(res)
}
