use clone_dyn::clone_dyn;

use crate::functions::rollup::TimeseriesMap;
use crate::functions::types::ParameterValue;

#[derive(Default, Clone)]
pub(crate) struct RollupFuncArg {
    /// The value preceding values if it fits staleness interval.
    pub(super) prev_value: f64,

    /// The timestamp for prev_value.
    pub(super) prev_timestamp: i64,

    /// Values that fit window ending at curr_timestamp.
    pub(crate) values: Vec<f64>,

    /// Timestamps for values.
    pub(crate) timestamps: Vec<i64>,

    /// Real value preceding values without restrictions on staleness interval.
    pub(super) real_prev_value: f64,

    /// Real value which goes after values.
    pub(crate) real_next_value: f64,

    /// Current timestamp for rollup evaluation.
    pub(super) curr_timestamp: i64,

    /// Index for the currently evaluated point relative to time range for query evaluation.
    pub(super) idx: usize,

    /// Time window for rollup calculations.
    pub(super) window: i64,

    pub(super) tsm: Option<TimeseriesMap>,
}

impl RollupFuncArg {
    pub fn reset(mut self) {
        self.prev_value = 0.0;
        self.prev_timestamp = 0;
        self.values = vec![];
        self.timestamps = vec![];
        self.curr_timestamp = 0;
        self.idx = 0;
        self.window = 0;
        if let Some(mut tsm) = self.tsm {
            tsm.reset()
        }
    }
}


pub(crate) type RollupFunc = fn(rfa: &mut RollupFuncArg) -> f64;

#[clone_dyn]
/// RollupFunc must return rollup value for the given rfa.
///
/// prev_value may be NAN, values and timestamps may be empty.
pub trait RollupFn: Fn(&mut RollupFuncArg) -> f64 + Send + Sync {}

/// implement `Rollup` on any type that implements `Fn(&RollupFuncArg) -> f64`.
impl<T> RollupFn for T where T: Fn(&mut RollupFuncArg) -> f64 + Send + Sync {}

#[clone_dyn]
pub trait NewRollupFn: Fn(&Vec<ParameterValue>) -> &Box<dyn RollupFn> {}

impl<T> NewRollupFn for T where T: Fn(&Vec<ParameterValue>) -> &Box<dyn RollupFn> {}

pub(super) type CountFilter = fn(values: &[f64], val: f64) -> i32;
