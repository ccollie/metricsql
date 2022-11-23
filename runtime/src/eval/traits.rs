use std::sync::Arc;

use crate::context::Context;
use crate::runtime_error::RuntimeResult;
use crate::{EvalConfig, QueryValue};

pub trait Engine {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue>;
}
