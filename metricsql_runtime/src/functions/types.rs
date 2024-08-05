use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::Timeseries;

#[inline]
pub(crate) fn get_single_timeseries(series: &Vec<Timeseries>) -> RuntimeResult<&Timeseries> {
    if series.len() != 1 {
        let msg = format!(
            "arg must contain a single timeseries; got {} timeseries",
            series.len()
        );
        return Err(RuntimeError::TypeCastError(msg));
    }
    Ok(&series[0])
}
