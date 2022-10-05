pub(crate) use aggr_fns::{
    AggrFn,
    AggrFuncArg,
    get_aggr_func
};
pub(crate) use aggr_incremental::{
    get_incremental_aggr_func_callbacks,
    IncrementalAggrFuncCallbacks,
    IncrementalAggrFuncContext
};

pub(self) mod aggr_fns;
mod aggr_incremental;
mod accumulator;

