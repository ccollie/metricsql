use futures::FutureExt;
use std::future::Future;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

// Code in this file is based on:
// https://github.com/dahomey-technologies/rustis/blob/main/src/network/async_excutor_strategy.rs
// License: MIT
use crate::{RuntimeError, RuntimeResult};

pub enum JoinHandle<T> {
    #[cfg(feature = "tokio-runtime")]
    Tokio(tokio::task::JoinHandle<T>),
    #[cfg(feature = "async-std-runtime")]
    AsyncStd(async_std::task::JoinHandle<T>),
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            #[cfg(feature = "tokio-runtime")]
            JoinHandle::Tokio(join_handle) => match join_handle.poll_unpin(cx) {
                Poll::Ready(Ok(result)) => Poll::Ready(Ok(result)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(RuntimeError::SpawnError(format!(
                    "Async JoinError: {e}"
                )))),
                Poll::Pending => Poll::Pending,
            },
            #[cfg(feature = "async-std-runtime")]
            JoinHandle::AsyncStd(join_handle) => match join_handle.poll_unpin(cx) {
                Poll::Ready(result) => Poll::Ready(Ok(result)),
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

pub fn spawn<F, T>(future: F) -> JoinHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    #[cfg(feature = "tokio-runtime")]
    return JoinHandle::Tokio(tokio::spawn(future));
    #[cfg(feature = "async-std-runtime")]
    JoinHandle::AsyncStd(async_std::task::spawn(future))
}

pub fn block_on_with_timeout<F, T>(timeout_duration: Duration, future: F) -> Result<T, RuntimeError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    if timeout_duration.is_zero() {
        return block_on(future);
    }
    futures::executor::block_on(async move { timeout(timeout_duration, future).await })
}

pub fn block_on<F>(future: F) -> RuntimeResult<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    // see https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust#solve-the-problem
    // https://github.com/tokio-rs/tokio/issues/237
    futures::executor::block_on(async move { spawn(future).await })
}

/// Await on a future for a maximum amount of time before returning an error.
#[allow(dead_code)]
pub(crate) async fn timeout<F: Future>(
    timeout: Duration,
    future: F,
) -> Result<F::Output, RuntimeError> {
    #[cfg(feature = "tokio-runtime")]
    {
        tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| RuntimeError::Timeout("The I/O operation’s timeout expired".to_owned()))
    }
    #[cfg(feature = "async-std-runtime")]
    {
        // This avoids a panic on async-std when the provided duration is too large.
        // See: https://github.com/async-rs/async-std/issues/1037.
        if timeout == Duration::MAX {
            Ok(future.await)
        } else {
            async_std::future::timeout(timeout, future)
                .await
                .map_err(|_| {
                    RuntimeError::Timeout("The I/O operation’s timeout expired".to_owned())
                })
        }
    }
}
