#[cfg(feature = "async_std")]
use async_executors::AsyncStd;
#[cfg(feature = "async_std")]
pub use AsyncStd as DefaultAsyncHandler;
#[cfg(feature = "async_std")]
pub use AsyncStd as AsyncStdHandler;

#[cfg(feature = "async_std")]
pub(crate) fn create_default_handler() -> impl AsyncHandler {
    AsyncStd::new()
}
