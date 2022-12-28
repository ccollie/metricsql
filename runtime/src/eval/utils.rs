use crate::functions::types::AnyValue;

pub(crate) fn series_len(val: &AnyValue) -> usize {
    match &val {
        AnyValue::RangeVector(iv) |
        AnyValue::InstantVector(iv) => iv.len(),
        _ => {
            1
        }
    }
}