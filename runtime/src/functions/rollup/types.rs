use crate::types::Timestamp;
use crate::{QueryValue, RuntimeResult};
use clone_dyn::clone_dyn;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use crate::functions::rollup::TimeseriesMap;

#[derive(Default, Clone, Debug)]
pub struct RollupFuncArg {
    /// The value preceding values if it fits staleness interval.
    pub(super) prev_value: f64,

    /// The timestamp for prev_value.
    pub(super) prev_timestamp: Timestamp,

    /// Values that fit window ending at curr_timestamp.
    pub(crate) values: Vec<f64>,

    /// Timestamps for values.
    pub(crate) timestamps: Vec<i64>,

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

    pub(super) tsm: Option<Rc<RefCell<TimeseriesMap>>>,
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
        if let Some(tsm) = self.tsm {
            tsm.borrow_mut().reset()
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
pub(crate) trait NewRollupFn: Fn(&Vec<QueryValue>) -> Arc<dyn RollupFn> {}

impl<T> NewRollupFn for T where T: Fn(&Vec<QueryValue>) -> Arc<dyn RollupFn> {}

pub(crate) trait RollupHandler {
    fn init(&mut self, _args: &[QueryValue]) {}
    fn eval(&self, arg: &mut RollupFuncArg) -> f64;
}

impl<F> RollupHandler for F
where
    F: Fn(&mut RollupFuncArg) -> f64 + Send + Sync,
{
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        self(arg)
    }
}

impl Debug for dyn RollupHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn RollupHandler")
    }
}

/// Wrapper over raw functions
pub(crate) struct RawRollupHandler(RollupFunc);

impl RollupHandler for RawRollupHandler {
    #[inline]
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        (self.0)(arg)
    }
}

impl Debug for RawRollupHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RollupHandler")
    }
}

pub(crate) struct FakeRollupHandler(&'static str);

impl RollupHandler for FakeRollupHandler {
    fn eval(&self, _: &mut RollupFuncArg) -> f64 {
        panic!("BUG: RollupHandler '{}' shouldn't be called", self.0);
    }
}

#[derive(Clone)]
pub(crate) enum RollupHandlerEnum {
    Wrapped(RollupFunc),
    Fake(&'static str),
    General(Box<dyn RollupFn<Output = f64>>),
}

impl RollupHandlerEnum {
    pub fn wrap(f: RollupFunc) -> Self {
        RollupHandlerEnum::Wrapped(f)
    }
    pub fn fake(name: &'static str) -> Self {
        RollupHandlerEnum::Fake(name)
    }
}

impl RollupHandler for RollupHandlerEnum {
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        match self {
            RollupHandlerEnum::Wrapped(wrapped) => wrapped(arg),
            RollupHandlerEnum::Fake(name) => {
                panic!("BUG: {} shouldn't be called", name);
            }
            RollupHandlerEnum::General(df) => df(arg),
        }
    }
}

pub(crate) type RollupHandlerFactory = fn(&Vec<QueryValue>) -> RuntimeResult<RollupHandlerEnum>;
