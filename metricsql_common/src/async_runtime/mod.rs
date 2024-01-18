use std::future::Future;
use std::sync::OnceLock;
use agnostic::Runtime;
use std::time::Duration;


cfg_if! {
    if #[cfg(feature = "tokio")] {
        pub type AsyncRuntime = agnostic::tokio::TokioRuntime;
     } else if #[cfg(feature = "async-std")] {
        pub type AsyncRuntime = agnostic::async_std::SmolRuntime;
     } else if #[cfg(feature = "smol")] {
        pub type AsyncRuntime = agnostic::smol::TokioRuntime;
     } else {
        unimplemented!("No async runtime feature enabled");
     }
}

pub type JoinHandle<T> = <AsyncRuntime as Runtime>::JoinHandle<T>;
pub type Timeout<F> = <AsyncRuntime as Runtime>::Timeout<F>;

pub fn get_runtime() -> &'static AsyncRuntime {
    static RUNTIME: OnceLock<AsyncRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| AsyncRuntime::new())
}

pub fn runtime() -> &'static impl Runtime {
    get_runtime()
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