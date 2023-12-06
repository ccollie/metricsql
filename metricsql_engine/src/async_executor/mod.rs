#[cfg(feature = "async-std")]
mod async_std;
#[cfg(feature = "tokio")]
mod tokio;

pub trait AsyncExecutor {
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output;
}
