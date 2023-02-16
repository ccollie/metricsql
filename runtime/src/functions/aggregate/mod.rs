pub(crate) use aggr_fns::{get_aggr_func, AggrFn, AggrFuncArg};
pub(crate) use aggr_incremental::{
    get_incremental_aggr_func_callbacks, IncrementalAggrFuncCallbacks, IncrementalAggrFuncContext,
};

mod accumulator;
pub(self) mod aggr_fns;
mod aggr_incremental;

#[cfg(test)]
mod aggr_incremental_test;
#[cfg(test)]
mod aggr_test;
