extern crate byte_pool;
extern crate byte_slice_cast;
extern crate core;
extern crate integer_encoding;
extern crate lockfree_object_pool;
extern crate lz4_flex;
extern crate once_cell;
extern crate q_compress;
extern crate rand;
#[cfg(feature = "xxh64")]
extern crate xxhash_rust;

mod atomic_counter;
mod bits;
mod decimal;
mod dedup;
mod duration;
mod encoding;
pub mod error;
mod fast_cache;
mod fastnum;
mod math;
mod no_hash;
mod pool;
mod random;
mod time;

pub use atomic_counter::*;
pub use bits::*;
pub use decimal::*;
pub use dedup::{deduplicate_samples, deduplicate_samples_during_merge};
pub use duration::*;
pub use encoding::*;
pub use fast_cache::*;
pub use fastnum::*;
pub use math::*;
pub use pool::*;
pub use random::*;
pub use time::*;
pub use no_hash::{IntMap, IntSet, NoHashHasher, BuildNoHashHasher};

#[cfg(test)]
pub(crate) mod tests;
