use std::any::Any;
use std::future::Future;
use std::sync::OnceLock;
use agnostic::Runtime;
use std::time::Duration;
use snafu::Snafu;
use crate::error::{ErrorExt, StatusCode};

pub use futures::future::join_all;
pub use futures::future::try_join_all;

cfg_if! {
    if #[cfg(feature = "tokio")] {
        pub type AsyncRuntime = agnostic::tokio::TokioRuntime;
     } else if #[cfg(feature = "async-std")] {
        pub type AsyncRuntime = agnostic::async_std::SmolRuntime;
     } else if #[cfg(feature = "smol")] {
        pub type AsyncRuntime = agnostic::smol::SmolRuntime;
     } else {
        unimplemented!("No async runtime feature enabled");
     }
}

pub type JoinHandle<T> = <AsyncRuntime as Runtime>::JoinHandle<T>;
pub type Timeout<F> = <AsyncRuntime as Runtime>::Timeout<F>;

pub type Result<T> = std::result::Result<T, Error>;
pub type AsyncRuntimeResult<T> = std::result::Result<T, Error>;

/// Default error implementation of table.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("Join error: {}", msg))]
    Join {
        msg: String,
    },
    #[snafu(display("Async operation timed out after {:?}", duration))]
    Timeout {
        duration: Duration
    },
    #[snafu(display("Unexpected error"))]
    Execution {
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}
impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;
        match self {
            Join { .. } => StatusCode::Internal,
            Timeout { .. } => StatusCode::TimedOut,
            Execution { .. } => StatusCode::Unexpected
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn get_runtime() -> &'static AsyncRuntime {
    static RUNTIME: OnceLock<AsyncRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| AsyncRuntime::new())
}

pub fn runtime() -> &'static impl Runtime {
    get_runtime()
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    AsyncRuntime::block_on(future)
}


// Call an async function from a sync context
// see https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust#solve-the-problem
pub fn block_sync<F>(future: F) -> Result<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
{
    let res = futures::executor::block_on(async {
        spawn(future).await
    });
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(Error::Join { msg: e.to_string() }),
    }
}

pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
{
    AsyncRuntime::spawn(future)
}

pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static {

    AsyncRuntime::spawn_blocking(f)
}

pub fn timeout<F>(duration: Duration, future: F) -> Timeout<F>
    where
        F: Future + Send
{
    AsyncRuntime::timeout(duration, future)
}