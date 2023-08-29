use std::fmt::Debug;
use std::sync::Arc;

use clone_dyn::clone_dyn;
use tinyvec::TinyVec;

use crate::functions::rollup::TimeseriesMap;
use crate::types::Timestamp;
use crate::{QueryValue, RuntimeResult};

#[derive(Default, Clone, Debug)]
pub struct RollupFuncArg<'a> {
    /// The value preceding values if it fits staleness interval.
    pub(super) prev_value: f64,

    /// The timestamp for prev_value.
    pub(super) prev_timestamp: Timestamp,

    /// Values that fit window ending at curr_timestamp.
    pub(crate) values: &'a [f64],

    /// Timestamps for values.
    pub(crate) timestamps: &'a [i64],

    /// Real value preceding values without restrictions on staleness interval.
    pub(super) real_prev_value: f64,

    /// Real value which goes after values.
    pub(crate) real_next_value: f64,

    /// Current timestamp for rollup evaluation.
    pub(super) curr_timestamp: Timestamp,

    /// Index for the currently evaluated point relative to time range for query evaluation.
    pub(super) idx: usize,

    /// Time window for rollup calculations.
    pub(super) window: i64,

    pub(super) tsm: Option<Arc<TimeseriesMap>>,
}

static EMPTY_TIMESTAMPS: &[f64] = &[];

impl<'a> RollupFuncArg<'a> {
    pub(crate) fn get_tsm(&self) -> Arc<TimeseriesMap> {
        if let Some(tsm) = &self.tsm {
            tsm.clone()
        } else {
            panic!("BUG: tsm is None")
        }
    }
}

pub(crate) type RollupFunc = fn(rfa: &RollupFuncArg) -> f64;

#[clone_dyn]
/// RollupFunc must return rollup value for the given rfa.
///
/// prev_value may be NAN, values and timestamps may be empty.
pub trait RollupFn: Fn(&RollupFuncArg) -> f64 + Send + Sync {}

/// implement `Rollup` on any type that implements `Fn(&RollupFuncArg) -> f64`.
impl<T> RollupFn for T where T: Fn(&RollupFuncArg) -> f64 + Send + Sync {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GenericRollupHandler<T, F>
where
    F: Fn(&RollupFuncArg, &T) -> f64 + Send + Sync,
    T: Clone + Debug,
{
    pub(crate) state: T,
    pub(crate) func: F,
}

impl<T, F> GenericRollupHandler<T, F>
where
    F: Fn(&RollupFuncArg, &T) -> f64 + Send + Sync,
    T: Clone + Debug,
{
    pub fn new(state: T, func: F) -> Self {
        Self { state, func }
    }

    pub(crate) fn eval(&self, arg: &RollupFuncArg) -> f64 {
        (self.func)(arg, &self.state)
    }
}

pub(crate) type RollupHandlerFloatArg = GenericRollupHandler<f64, fn(&RollupFuncArg, &f64) -> f64>;

pub(crate) type RollupHandlerVecArg =
    GenericRollupHandler<TinyVec<[f64; 4]>, fn(&RollupFuncArg, &TinyVec<[f64; 4]>) -> f64>;

#[derive(Clone, Debug)]
pub(crate) enum RollupHandler {
    Wrapped(RollupFunc),
    Fake(&'static str),
    FloatArg(RollupHandlerFloatArg),
    VecArg(RollupHandlerVecArg),
    General(Box<dyn RollupFn<Output = f64>>),
}

impl Debug for dyn RollupFn<Output = f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn RollupFn")
    }
}

impl RollupHandler {
    pub fn wrap(f: RollupFunc) -> Self {
        RollupHandler::Wrapped(f)
    }
    pub fn fake(name: &'static str) -> Self {
        RollupHandler::Fake(name)
    }

    pub(crate) fn eval(&self, arg: &RollupFuncArg) -> f64 {
        match self {
            RollupHandler::Wrapped(wrapped) => wrapped(arg),
            RollupHandler::Fake(name) => {
                panic!("BUG: {} shouldn't be called", name);
            }
            RollupHandler::General(df) => df(arg),
            RollupHandler::FloatArg(f) => f.eval(arg),
            RollupHandler::VecArg(f) => f.eval(arg),
        }
    }
}

impl PartialEq for RollupHandler {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RollupHandler::Wrapped(left), RollupHandler::Wrapped(right)) => left == right,
            (RollupHandler::Fake(left), RollupHandler::Fake(right)) => left == right,
            (RollupHandler::FloatArg(left), RollupHandler::FloatArg(right)) => left == right,
            (RollupHandler::VecArg(left), RollupHandler::VecArg(right)) => left == right,
            (RollupHandler::General(_), RollupHandler::General(_)) => false,
            _ => false,
        }
    }
}

impl Default for RollupHandler {
    fn default() -> Self {
        RollupHandler::fake("default")
    }
}

pub(crate) type RollupHandlerFactory = fn(&Vec<QueryValue>) -> RuntimeResult<RollupHandler>;
