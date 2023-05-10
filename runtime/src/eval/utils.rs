
use crate::{QueryValue, Timeseries};

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
