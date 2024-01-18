#[cfg(feature = "tokio")]
use agnostic::tokio::TokioRuntime;
#[cfg(feature = "smol")]
use agnostic::smol::SmolRuntime;


fn create_runtime() -> impl Runtime {
    #[cfg(feature = "tokio")]
    return TokioRuntime::new();

}