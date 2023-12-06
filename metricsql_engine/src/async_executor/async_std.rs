use crate::async_executor::AsyncExecutor;

pub struct AsyncStdExecutor {}

impl AsyncExecutor for AsyncStdExecutor {
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        async_std::task::block_on(future)
    }
}
