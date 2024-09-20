#[cfg(target_arch = "aarch64")]
pub use gxhash::*;
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
pub use gxhash::*;

#[cfg(not(any(
    all(target_feature = "avx2", target_arch = "x86_64"),
    target_arch = "aarch64"
)))]
pub use fast_hash_fallback::*;
pub use no_hash::*;

#[cfg(target_arch = "aarch64")]
mod gxhash;

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
mod gxhash;

#[cfg(not(any(
    all(target_feature = "avx2", target_arch = "x86_64"),
    target_arch = "aarch64"
)))]
mod fast_hash_fallback;

mod no_hash;
mod signature;

pub use signature::*;