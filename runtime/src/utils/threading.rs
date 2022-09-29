use rayon::iter::{IntoParallelIterator, ParallelIterator};
use crate::runtime_error::RuntimeResult;

/// run_parallel runs f in parallel for all the results from rss.
pub(crate) fn run_parallel<T, F, P>(data: &[T], param: P, f: F) -> RuntimeResult<()>
    where F: Fn(&T) -> RuntimeResult<()> + Send + Sync {

    // data.into_par_iter().try_for_each(|t|  {
    //     f(t)
    // })

    todo!()
}