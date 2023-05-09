use metricsql::ast::BinaryExpr;
use metricsql::common::Operator;
use crate::{QueryValue, Timeseries};

pub(crate) fn series_len(val: &QueryValue) -> usize {
    match &val {
        QueryValue::RangeVector(iv) | QueryValue::InstantVector(iv) => iv.len(),
        _ => 1,
    }
}