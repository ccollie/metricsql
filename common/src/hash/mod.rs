use cfg_if::cfg_if;

cfg_if!(
    if #[cfg(all(feature = "gxhash", target_feature="aes"))] {
        pub use gx_hash::*;
    } else {
        mod fast_hash_fallback;
        pub use fast_hash_fallback::*;
    }
);

mod gx_hash;
mod no_hash;
mod signature;

pub use no_hash::*;
pub use signature::*;
