use crate::execution::remove_empty_series;
use crate::functions::transform::TransformFuncArg;
use crate::{types::Timeseries, RuntimeResult};

pub(crate) fn limit_offset(tfa: &mut TransformFuncArg) -> RuntimeResult<Vec<Timeseries>> {
    let limit = tfa.get_param_usize(0, "limit")?;
    let offset = tfa.get_param_usize(1, "offset")?;
    let mut rvs = tfa.get_param_series(2)?;

    // remove_empty_series so offset will be calculated after empty series
    // were filtered out.
    remove_empty_series(&mut rvs);

    if rvs.len() >= offset {
        if offset > 0 {
            rvs.drain(0..offset);
        }
    } else {
        return Ok(vec![]);
    }
    if rvs.len() > limit {
        rvs.truncate(limit);
    }

    Ok(rvs)
}
