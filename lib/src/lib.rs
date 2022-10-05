extern crate byte_pool;
extern crate byte_slice_cast;
extern crate instant;
extern crate integer_encoding;
extern crate lockfree_object_pool;
extern crate lz4_flex;
extern crate once_cell;
extern crate q_compress;
extern crate rand;
#[cfg(feature = "xxh64")]
extern crate xxhash_rust;
extern crate core;

mod bits;
mod decimal;
mod encoding;
pub mod error;
mod fastnum;
mod math;
mod pool;
mod random;
mod time;
mod fast_cache;
mod dedup;

pub use bits::*;
pub use decimal::*;
pub use encoding::*;
pub use fastnum::*;
pub use math::*;
pub use pool::*;
pub use random::*;
pub use time::*;
pub use fast_cache::*;
pub use dedup::{ deduplicate_samples, deduplicate_samples_during_merge };
