use byte_pool::{Block, BytePool};
use std::sync::OnceLock;

static BYTE_POOL: OnceLock<BytePool<Vec<u8>>> = OnceLock::new();

pub fn get_pooled_buffer(size: usize) -> Block<'static, Vec<u8>> {
    let pool = BYTE_POOL.get_or_init(BytePool::new);
    let mut buf = pool.alloc(size);
    buf.clear();
    buf
}

static INT64_POOL: OnceLock<BytePool<Vec<i64>>> = OnceLock::new();

/// get_int64s returns an int64 slice with the given size.
pub fn get_int64s(size: usize) -> Block<'static, Vec<i64>> {
    let pool = INT64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v
}

static F64_POOL: OnceLock<BytePool<Vec<f64>>> = OnceLock::new();

/// get_int64s returns an int64 slice with the given size.
pub fn get_float64s(size: usize) -> Block<'static, Vec<f64>> {
    let pool = F64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v
}
