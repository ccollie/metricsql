use ahash::{AHashSet, AHasher};
use metricsql_parser::prelude::{BinaryExpr, Expr, Matcher, Operator};
use regex::escape;
use small_map::SmallMap;
use std::hash::BuildHasherDefault;
use crate::prelude::InstantVector;
use crate::{RuntimeError, RuntimeResult};
use crate::types::{Label, QueryValue, Timeseries};

pub(crate) fn can_push_down_common_filters(be: &BinaryExpr) -> bool {
    if be.op == Operator::Or || be.op == Operator::Default {
        return false;
    }
    match (&be.left.as_ref(), &be.right.as_ref()) {
        (Expr::Aggregation(left), Expr::Aggregation(right)) => {
            if left.is_non_grouping() || right.is_non_grouping() {
                return false;
            }
            true
        }
        _ => true,
    }
}

pub(crate) fn get_common_label_filters(tss: &[Timeseries]) -> Vec<Matcher> {
    let mut kv_map: SmallMap<16, String, AHashSet<String>, BuildHasherDefault<AHasher>> =
        SmallMap::new();
    for ts in tss.iter() {
        for Label { name: k, value: v } in ts.metric_name.labels.iter() {
            match kv_map.get_mut(k) {
                Some(set) => {
                    set.insert(v.to_string());
                }
                None => {
                    let mut set = AHashSet::with_capacity(8);
                    set.insert(v.to_string());
                    kv_map.insert(k.to_string(), set);
                }
            }
        }
    }

    let mut lfs: Vec<Matcher> = Vec::with_capacity(kv_map.len());
    for (key, values) in kv_map {
        if values.len() != tss.len() {
            // Skip the tag, since it doesn't belong to all the time series.
            continue;
        }

        if values.len() > 1000 {
            // Skip the filter on the given tag, since it needs to enumerate too many unique values.
            // This may slow down the provider for matching time series.
            continue;
        }

        let vals: Vec<&String> = values.iter().collect::<Vec<_>>();

        let lf = if values.len() == 1 {
            Matcher::equal(key, vals[0].into())
        } else {
            let str_value = join_regexp_values(&vals);
            Matcher::regex_equal(key, str_value).unwrap()
        };

        lfs.push(lf);
    }
    // todo(perf): does this need to be sorted ?
    lfs.sort();
    lfs
}

fn join_regexp_values(a: &[&String]) -> String {
    let init_size = a.iter().fold(0, |res, x| res + x.len() + 3);
    let mut res = String::with_capacity(init_size);
    for (i, s) in a.iter().enumerate() {
        let s_quoted = escape(s);
        res.push_str(s_quoted.as_str());
        if i < a.len() - 1 {
            res.push('|')
        }
    }
    res
}

pub(super) fn contains_value_at(tss: &[Timeseries], v: f64, idx: usize) -> bool {
    tss.iter().any(|ts| ts.values[idx] == v)
}

pub(super) fn not_contains_value_at(tss: &[Timeseries], v: f64, idx: usize) -> bool {
    !contains_value_at(tss, v, idx)
}

pub(super) fn handle_vector_scalar_list_equality(
    vector: InstantVector,
    scalar: f64,
    op: Operator,
) -> RuntimeResult<QueryValue> {
    let mut vector = vector;

    if op == Operator::Eql || op == Operator::NotEq {
        // scalar != (1,2,3) or scalar == (1,2,3)
        let is_equals = op == Operator::Eql;

        for ts in vector.iter_mut() {
            if is_equals {
                for v in ts.values.iter_mut().filter(|val| *val != &scalar) {
                    *v = f64::NAN;
                }
            } else {
                for v in ts.values.iter_mut().filter(|val| *val == &scalar) {
                    *v = f64::NAN;
                }
            }
        }
    } else {
        return Err(RuntimeError::ArgumentError(
            "expected equality or inequality operator for scalar vector list comparison".to_string(),
        ));
    }

    Ok(QueryValue::InstantVector(vector))
}