use crate::functions::rollup::{RollupFn, RollupFunc, RollupFuncArg};
use crate::functions::types::ParameterValue;

pub(crate) trait RollupHandler {
    fn eval(&self, arg: &mut RollupFuncArg) -> f64;
}

pub(crate) enum RollupHandlerEnum {
    Wrapped(RollupFunc),
    Fake(&'static str),
    General(Box<dyn RollupFn>),
}

impl RollupHandlerEnum {
    pub fn wrap(f: RollupFunc) -> Self {
        RollupHandlerEnum::Wrapped(f)
    }

    pub fn fake(name: &'static str) -> Self {
        RollupHandlerEnum::Fake(name)
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
            RollupHandlerEnum::Fake(name) => {
                panic!("BUG: {} shouldn't be called", name);
            },
            RollupHandlerEnum::General(df) => df(arg)
        }
    }
}

pub(crate) type RollupHandlerFactory = fn(&Vec<ParameterValue>) -> RollupHandlerEnum;