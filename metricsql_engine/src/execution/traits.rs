use crate::execution::{Context, EvalConfig};
use crate::runtime_error::RuntimeResult;
use crate::QueryValue;

pub trait Engine {
    fn eval(&self, ctx: &Context, ec: &EvalConfig) -> RuntimeResult<QueryValue>;
}
