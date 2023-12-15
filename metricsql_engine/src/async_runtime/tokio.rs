#[cfg(feature = "tokio_ct")]
use async_executors::TokioCt;
#[cfg(any(feature = "tokio_tp"))]
use async_executors::TokioTp;

#[cfg(any(feature = "tokio_tp", feature = "tokio_ct"))]
use super::AsyncHandler;

#[cfg(any(feature = "tokio_tp"))]
pub use TokioTp as TokioTpHandler;

#[cfg(any(feature = "tokio_tp"))]
pub(crate) fn create_default_handler() -> impl AsyncHandler {
    TokioTp::try_current().unwrap()
}

#[cfg(feature = "tokio_ct")]
pub use TokioCt as TokioCurrentThreadHandler;

#[cfg(feature = "tokio_ct")]
pub(crate) fn create_default_handler() -> impl AsyncHandler {
    TokioCt::new().unwrap()
}
