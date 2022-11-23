use std::sync::OnceLock;

use byte_pool::{Block, BytePool};

static BYTE_POOL: OnceLock<BytePool<Vec<u8>>> = OnceLock::new();

pub fn get_pooled_buffer(size: usize) -> Block<'static, Vec<u8>> {
    let pool = BYTE_POOL.get_or_init(BytePool::new);
    let mut buf = pool.alloc(size);
    buf.clear();
    buf
}

static INT64_POOL: OnceLock<BytePool<Vec<i64>>> = OnceLock::new();

/// get_pooled_vec_i64 returns an int64 slice with the given size.
pub fn get_pooled_vec_i64(size: usize) -> Block<'static, Vec<i64>> {
    let pool = INT64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v.reserve(size);
    v
}

/// get_pool_vec_i64_filled returns an int64 slice with the given size and filled with the given value.
pub fn get_pool_vec_i64_filled(size: usize, value: i64) -> Block<'static, Vec<i64>> {
    let mut v = get_pooled_vec_i64(size);
    v.resize_with(size, || value);
    v
}

static F64_POOL: OnceLock<BytePool<Vec<f64>>> = OnceLock::new();

/// get_pooled_vec_f64 returns an f64 slice with the given size.
pub fn get_pooled_vec_f64(size: usize) -> Block<'static, Vec<f64>> {
    let pool = F64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v.reserve(size);
    v
}

/// get_pooled_vec_f64_filled returns an f64 slice with the given size and filled with the given value.
pub fn get_pooled_vec_f64_filled(size: usize, value: f64) -> Block<'static, Vec<f64>> {
    let mut v = get_pooled_vec_f64(size);
    v.resize_with(size, || value);
    v
}
