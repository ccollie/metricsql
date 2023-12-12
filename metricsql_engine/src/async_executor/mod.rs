use std::future::Future;

use agnostik::{executor, AgnostikExecutor};

pub fn get_runtime() -> &'static impl AgnostikExecutor {
    executor()
}

// see https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust#solve-the-problem
// https://github.com/tokio-rs/tokio/issues/2376

pub fn block_on<F>(future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    get_runtime().block_on(future)
}
