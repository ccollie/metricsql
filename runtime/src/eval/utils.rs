use std::collections::HashMap;
use crate::{Labels, QueryValue, Timeseries};
use crate::signature::Signature;

pub(crate) fn series_len(val: &QueryValue) -> usize {
    match &val {
        QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => iv.len(),
        _ => 1,
    }
}

#[inline]
pub fn remove_empty_series(tss: &mut Vec<Timeseries>) {
    tss.retain(|ts| !ts.values.iter().all(|v| v.is_nan()));
}

pub(crate) struct ArithmeticItem {
    pub(crate) labels: Labels,
    pub(crate) value: f64,
    pub(crate) num: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TopItem {
    pub(crate) index: usize,
    pub(crate) value: f64,
}

pub(crate) fn eval_arithmetic(
    param: &Option<LabelModifier>,
    data: &Value,
    f_name: &str,
    f_handler: fn(total: f64, val: f64) -> f64,
) -> Result<Option<HashMap<Signature, ArithmeticItem>>> {
    let data = match data {
        Value::Vector(v) => v,
        Value::None => return Ok(None),
        _ => {
            return Err(DataFusionError::Plan(format!(
                "[{f_name}] function only accept vector values"
            )))
        }
    };

    let mut score_values = HashMap::default();
    match param {
        Some(v) => match v {
            LabelModifier::Include(labels) => {
                for item in data.iter() {
                    let mut sum_labels = Labels::default();
                    for label in item.labels.iter() {
                        if labels.contains(&label.name) {
                            sum_labels.push(label.clone());
                        }
                    }
                    let sum_hash = signature(&sum_labels);
                    let entry = score_values.entry(sum_hash).or_insert(ArithmeticItem {
                        labels: sum_labels,
                        value: 0.0,
                        num: 0,
                    });
                    entry.value = f_handler(entry.value, item.sample.value);
                    entry.num += 1;
                }
            }
            LabelModifier::Exclude(labels) => {
                for item in data.iter() {
                    let mut sum_labels = Labels::default();
                    for label in item.labels.iter() {
                        if !labels.contains(&label.name) {
                            sum_labels.push(label.clone());
                        }
                    }
                    let sum_hash = signature(&sum_labels);
                    let entry = score_values.entry(sum_hash).or_insert(ArithmeticItem {
                        labels: sum_labels,
                        value: 0.0,
                        num: 0,
                    });
                    entry.value = f_handler(entry.value, item.sample.value);
                    entry.num += 1;
                }
            }
        },
        None => {
            for item in data.iter() {
                let entry = score_values
                    .entry(Signature::default())
                    .or_insert(ArithmeticItem {
                        labels: Labels::default(),
                        value: 0.0,
                        num: 0,
                    });
                entry.value = f_handler(entry.value, item.sample.value);
                entry.num += 1;
            }
        }
    }
    Ok(Some(score_values))
}
