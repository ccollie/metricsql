use crate::functions::rollup::{RollupFn, RollupFunc, RollupFuncArg};
use crate::functions::types::ParameterValue;

pub(crate) trait RollupHandler {
    fn eval(&self, arg: &mut RollupFuncArg) -> f64;
}

pub(crate) trait StatefulRollupHandler: RollupHandler {
    /// State type
    type S;

    fn state(&self) -> &Self::S;
}

pub(crate) struct FakeHandler {
    name: &'static str,
}

impl FakeHandler {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl RollupHandler for FakeHandler {
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        panic!("BUG: {} shouldn't be called", self.name);
    }
}


/// Wrapper for state based configurable functions
pub(crate) struct GenericHandler<S, F>
    where
        F: Fn(&S, &mut RollupFuncArg) -> f64
{
    pub state: S,
    _exec: F,
}

impl<S, F> GenericHandler<S, F>
    where
        F: Fn(&S, &mut RollupFuncArg) -> f64
{
    pub fn new(state: S, exec: F) -> Self {
        Self {
            state,
            _exec: exec,
        }
    }
}

impl<S, F> RollupHandler for GenericHandler<S, F>
    where
        F: Fn(&S, &mut RollupFuncArg) -> f64
{
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        (self._exec)(&self.state, arg)
    }
}

pub(crate) enum RollupHandlerEnum {
    Wrapped(RollupFunc),
    Fake(FakeHandler),
    General(Box<dyn RollupFn>),
}

impl RollupHandlerEnum {
    pub fn wrap(f: RollupFunc) -> Self {
        RollupHandlerEnum::Wrapped(f)
    }

    pub fn fake(name: &'static str) -> Self {
        RollupHandlerEnum::Fake(FakeHandler::new(name))
    }

    pub fn is_wrapped(&self) -> bool {
        match self {
            RollupHandlerEnum::Wrapped(_) => true,
            _ => false
        }
    }
}

impl RollupHandler for RollupHandlerEnum {
    fn eval(&self, arg: &mut RollupFuncArg) -> f64 {
        match self {
            RollupHandlerEnum::Wrapped(wrapped) => wrapped(arg),
            RollupHandlerEnum::Fake(f) => f.eval(arg),
            RollupHandlerEnum::General(df) => df(arg)
        }
    }
}

pub(crate) type RollupHandlerFactory = fn(&Vec<ParameterValue>) -> RollupHandlerEnum;