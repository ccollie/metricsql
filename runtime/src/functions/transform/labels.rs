use std::borrow::Cow;
use std::collections::HashMap;

use regex::Regex;

use metricsql::parser::compile_regexp;

use crate::functions::arg_parse::{get_series_arg, get_string_arg};
use crate::functions::transform::TransformFuncArg;
use crate::{MetricName, RuntimeError, RuntimeResult, Timeseries, METRIC_NAME_LABEL};

const DOT_SEPARATOR: &str = ".";

pub(crate) fn label_keep(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut keep_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let keep_label = get_string_arg(&tfa.args, i)?;
        keep_labels.push(keep_label.to_string());
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        ts.metric_name.retain_tags(&keep_labels)
    }

    Ok(series)
}

pub(crate) fn label_del(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut del_labels: Vec<String> = Vec::with_capacity(tfa.args.len());
    for i in 1..tfa.args.len() {
        let del_label = get_string_arg(&tfa.args, i)?;
        del_labels.push(del_label.to_string());
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        ts.metric_name.remove_tags(&del_labels)
    }

    Ok(series)
}

pub(crate) fn label_set(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let (dst_labels, dst_values) = get_string_pairs(tfa, 1)?;
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    handle_label_set(&mut series, &dst_labels, &dst_values);

    Ok(series)
}

pub(crate) fn alias(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let alias = get_string_arg(&tfa.args, 1)?.to_string();
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    handle_label_set(&mut series, &[METRIC_NAME_LABEL.to_string()], &[alias]);

    Ok(series)
}

pub(crate) fn handle_label_set(
    series: &mut [Timeseries],
    dst_labels: &[String],
    dst_values: &[String],
) {
    for ts in series.iter_mut() {
        for (dst_label, value) in dst_labels.iter().zip(dst_values.iter()) {
            if value.is_empty() {
                ts.metric_name.remove_tag(dst_label);
            } else {
                ts.metric_name.set_tag(dst_label, value)
            }
        }
    }
}

pub(crate) fn label_uppercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_value_func(tfa, |x| x.to_uppercase())
}

pub(crate) fn label_lowercase(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_value_func(tfa, |x| x.to_lowercase())
}

fn transform_label_value_func(
    tfa: &mut TransformFuncArg,
    f: fn(arg: &str) -> String,
) -> RuntimeResult<Vec<Timeseries>> {
    let mut labels = Vec::with_capacity(tfa.args.len() - 1);
    for i in 1..tfa.args.len() {
        let label = get_string_arg(&tfa.args, i)?;
        labels.push(label);
    }
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        for label in labels.iter() {
            let dst_value = get_tag_value(&ts.metric_name, label);
            let transformed = &*f(&dst_value);

            if transformed.is_empty() {
                ts.metric_name.remove_tag(label);
            } else {
                ts.metric_name.set_tag(label, transformed);
            }
        }
    }

    Ok(series)
}

pub(crate) fn label_map(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label = get_label(tfa, "", 1)?.to_string();

    let (src_values, dst_values) = get_string_pairs(tfa, 2)?;
    let mut m: HashMap<&str, &str> = HashMap::with_capacity(src_values.len());
    for (i, src_value) in src_values.iter().enumerate() {
        m.insert(src_value, &dst_values[i]);
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        let mut dst_value = get_tag_value(&ts.metric_name, &label);
        if let Some(value) = m.get(dst_value.as_str()) {
            dst_value.push_str(value);
        }
        if dst_value.is_empty() {
            ts.metric_name.remove_tag(&label);
        } else {
            ts.metric_name.set_tag(&label, &dst_value);
        }
    }

    Ok(series)
}

pub(crate) fn drop_common_labels(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for i in 1..tfa.args.len() {
        let mut other = get_series_arg(&tfa.args, i, tfa.ec)?;
        series.append(&mut other);
    }

    let mut counts_map: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for ts in series.iter() {
        ts.metric_name.count_label_values(&mut counts_map);
    }

    let series_len = series.len();
    // m.iter().filter(|entry| entry.1)
    for (label_name, x) in counts_map.iter() {
        for count in x.values() {
            if *count != series_len {
                continue;
            }
            for ts in series.iter_mut() {
                ts.metric_name.remove_tag(label_name);
            }
        }
    }

    Ok(series)
}

pub(crate) fn label_copy(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, false)
}

pub(crate) fn label_move(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    transform_label_copy_ext(tfa, true)
}

fn transform_label_copy_ext(
    tfa: &mut TransformFuncArg,
    remove_src_labels: bool,
) -> RuntimeResult<Vec<Timeseries>> {
    let (src_labels, dst_labels) = get_string_pairs(tfa, 1)?;

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        for (src_label, dst_label) in src_labels.iter().zip(dst_labels.iter()) {
            let value = ts.metric_name.tag_value(src_label);
            if value.is_none() {
                continue;
            }
            let value = value.unwrap();
            if value.is_empty() {
                // do not remove destination label if the source label doesn't exist.
                continue;
            }

            // this is done because value is a live (owned) ref to data in ts.metric_name
            // because of this, there was an outstanding borrow
            let v = value.clone();
            ts.metric_name.set_tag(dst_label, &v);

            if remove_src_labels && src_label != dst_label {
                ts.metric_name.remove_tag(src_label)
            }
        }
    }

    Ok(series)
}

pub(crate) fn label_join(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let dst_label = get_string_arg(&tfa.args, 1)?;
    let separator = get_string_arg(&tfa.args, 2)?;

    // todo: user something like SmallVec/StaticVec/ArrayVec
    let mut src_labels: Vec<String> = Vec::with_capacity(tfa.args.len() - 3);
    for i in 3..tfa.args.len() {
        let src_label = get_string_arg(&tfa.args, i)?;
        src_labels.push(src_label.to_string());
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        let mut dst_value = get_tag_value(&ts.metric_name, &dst_label);
        // use some manner of string buffer

        dst_value.clear(); //??? test this

        for (j, src_label) in src_labels.iter().enumerate() {
            if let Some(src_value) = ts.metric_name.tag_value(src_label) {
                dst_value.push_str(src_value);
            }
            if j + 1 < src_labels.len() {
                dst_value.push_str(&separator)
            }
        }

        if dst_value.is_empty() {
            ts.metric_name.remove_tag(&dst_label);
        } else {
            ts.metric_name.set_tag(&dst_label, &dst_value);
        }
    }

    Ok(series)
}

pub(crate) fn label_transform(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label = get_string_arg(&tfa.args, 1)?;
    let regex = get_string_arg(&tfa.args, 2)?;
    let replacement = get_string_arg(&tfa.args, 3)?;

    // todo: would it be useful to use a cache ?
    let r = compile_regexp(&regex);
    match r {
        Ok(..) => {}
        Err(err) => {
            return Err(RuntimeError::from(format!(
                "cannot compile regex {}: {:?}",
                &regex, err,
            )));
        }
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    handle_label_replace(&mut series, &label, &r.unwrap(), &label, &replacement)
}

pub(crate) fn label_replace(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let regex = get_string_arg(&tfa.args, 4)?.to_string();

    process_anchored_regex(tfa, regex.as_str(), |tfa, r| {
        let dst_label = get_string_arg(&tfa.args, 1)?;
        let replacement = get_string_arg(&tfa.args, 2)?;
        let src_label = get_string_arg(&tfa.args, 3)?;
        let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

        handle_label_replace(&mut series, &src_label, r, &dst_label, &replacement)
    })
}

const EMPTY_STRING: &str = "";

fn handle_label_replace(
    tss: &mut Vec<Timeseries>,
    src_label: &str,
    r: &Regex,
    dst_label: &str,
    replacement: &str,
) -> RuntimeResult<Vec<Timeseries>> {
    for ts in tss.iter_mut() {
        let src_value = ts.metric_name.tag_value(src_label);

        // note: we can have a match-all regex like `.*` which will match an empty string
        let haystack = if let Some(v) = src_value {
            v
        } else {
            EMPTY_STRING
        };

        if !r.is_match(haystack) {
            continue;
        }

        let b = r.replace_all(haystack, replacement);
        if b.len() == 0 {
            ts.metric_name.remove_tag(dst_label)
        } else {
            // if we have borrowed src_value, we need to clone it to avoid holding
            // a borrowed ref to ts.metric_name
            match b {
                Cow::Borrowed(_) => {
                    let cloned = b.to_string();
                    ts.metric_name.set_tag(dst_label, &cloned);
                }
                Cow::Owned(owned) => {
                    ts.metric_name.set_tag(dst_label, &owned);
                }
            };
        }
    }

    Ok(std::mem::take(tss))
}

pub(crate) fn label_value(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_name = get_string_arg(&tfa.args, 1)?;

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
    for ts in series.iter_mut() {
        ts.metric_name.reset_metric_group();
        let v = match ts.metric_name.tag_value(&label_name) {
            Some(v) => match v.parse::<f64>() {
                Ok(v) => v,
                Err(..) => f64::NAN,
            },
            None => f64::NAN,
        };

        for val in ts.values.iter_mut() {
            if !val.is_nan() {
                *val = v;
            }
        }
    }

    // do not remove timeseries with only NaN values, so `default` could be applied to them:
    // label_value(q, "label") default 123
    Ok(std::mem::take(&mut series))
}

fn process_anchored_regex<F>(
    tfa: &mut TransformFuncArg,
    re: &str,
    handler: F,
) -> RuntimeResult<Vec<Timeseries>>
where
    F: Fn(&mut TransformFuncArg, &Regex) -> RuntimeResult<Vec<Timeseries>>,
{
    let anchored = format!("^(?:{})$", re);
    match Regex::new(&anchored) {
        Err(e) => Err(RuntimeError::from(format!(
            "cannot compile regexp {} : {}",
            &re, e
        ))),
        Ok(ref r) => handler(tfa, r),
    }
}

pub(crate) fn label_match(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_name = get_label(tfa, "", 1)?.to_string();
    let label_re = get_label(tfa, "regexp", 2)?.to_string();

    process_anchored_regex(tfa, &label_re, move |tfa, r| {
        let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
        series.retain(|ts| {
            if let Some(label_value) = ts.metric_name.tag_value(&label_name) {
                r.is_match(label_value)
            } else {
                false
            }
        });

        Ok(series)
    })
}

pub(crate) fn label_mismatch(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let label_re = get_label(tfa, "regexp", 2)?.to_string();

    process_anchored_regex(tfa, &label_re, |tfa, r| {
        let label_name = get_label(tfa, "", 1)?;
        let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;
        series.retain(|ts| {
            if let Some(label_value) = ts.metric_name.tag_value(&label_name) {
                !r.is_match(label_value)
            } else {
                false
            }
        });

        Ok(std::mem::take(&mut series))
    })
}

pub(crate) fn label_graphite_group(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let mut group_ids: Vec<i64> = Vec::with_capacity(tfa.args.len() - 1);
    let group_args = &tfa.args[1..];
    for (i, arg) in group_args.iter().enumerate() {
        match arg.get_int() {
            Ok(gid) => group_ids.push(gid),
            Err(e) => {
                let msg = format!("cannot get group name from arg #{}: {:?}", i + 1, e);
                return Err(RuntimeError::ArgumentError(msg));
            }
        }
    }

    let mut series = get_series_arg(&tfa.args, 0, tfa.ec)?;

    for ts in series.iter_mut() {
        let groups: Vec<&str> = ts.metric_name.metric_group.split(DOT_SEPARATOR).collect();
        let mut group_name: String =
            String::with_capacity(ts.metric_name.metric_group.len() + groups.len());
        for (j, group_id) in group_ids.iter().enumerate() {
            if *group_id >= 0_i64 && *group_id < (groups.len() as i64) {
                let idx = *group_id as usize;
                group_name.push_str(groups[idx]);
            }
            if j < group_ids.len() - 1 {
                group_name.push('.')
            }
        }
        ts.metric_name.metric_group = group_name
    }

    Ok(std::mem::take(&mut series))
}

fn get_label<'a>(
    tfa: &'a TransformFuncArg,
    name: &str,
    arg_num: usize,
) -> RuntimeResult<Cow<'a, String>> {
    get_string_arg(&tfa.args, arg_num).map_err(|e| {
        RuntimeError::ArgumentError(format!(
            "cannot get {} label name from arg #{}: {:?}",
            name, arg_num, e
        ))
    })
}

fn get_string_pairs(
    tfa: &mut TransformFuncArg,
    start: usize,
) -> RuntimeResult<(Vec<String>, Vec<String>)> {
    if start >= tfa.args.len() {
        return Err(RuntimeError::ArgumentError(format!(
            "expected at least {} args; got {}",
            start + 2,
            tfa.args.len()
        )));
    }
    let args = &tfa.args[start..];
    let arg_len = args.len();
    if arg_len % 2 != 0 {
        return Err(RuntimeError::ArgumentError(format!(
            "the number of string args must be even; got {arg_len}"
        )));
    }
    let result_len = arg_len / 2;
    let mut ks: Vec<String> = Vec::with_capacity(result_len);
    let mut vs: Vec<String> = Vec::with_capacity(result_len);
    let mut i = start;
    while i < arg_len {
        let k = get_string_arg(&tfa.args, i)?;
        ks.push(k.to_string());
        let v = get_string_arg(&tfa.args, i + 1)?;
        vs.push(v.to_string());
        i += 2;
    }

    Ok((ks, vs))
}

fn get_tag_value(mn: &MetricName, dst_label: &str) -> String {
    match mn.tag_value(dst_label) {
        Some(val) => val.to_owned(),
        None => "".to_string(),
    }
}
