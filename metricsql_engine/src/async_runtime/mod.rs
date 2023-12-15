use std::future::Future;

use async_executors::{SpawnHandle, SpawnHandleExt};

#[cfg(feature = "async_std")]
pub use async_std::*;
#[cfg(any(feature = "tokio_tp", feature = "tokio_ct"))]
pub use tokio::*;

use crate::{QueryResults, RuntimeResult};

mod async_std;
mod tokio;

pub trait AsyncHandler: SpawnHandle<RuntimeResult<QueryResults>> + Clone {}

pub fn block_on<F>(runtime: impl AsyncHandler, future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    // see https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust#solve-the-problem
    // https://github.com/tokio-rs/tokio/issues/237
    futures::executor::block_on(async move { runtime.spawn_handle(future).await })
}
