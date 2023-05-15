mod aggr_fns;
mod incremental;

pub(crate) use aggr_fns::{get_aggr_func, AggrFn, AggrFuncArg};
pub(crate) use incremental::*;

#[cfg(test)]
mod aggr_incremental_test;
#[cfg(test)]
mod aggr_test;
