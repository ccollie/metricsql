use std::sync::{LazyLock, OnceLock};

use byte_pool::{Block, BytePool};

static BYTE_POOL: LazyLock<BytePool<Vec<u8>>> = LazyLock::new(BytePool::new);

pub type PooledBuffer = Block<'static, Vec<u8>>;
pub type PooledVecI64 = Block<'static, Vec<i64>>;
pub type PooledVecF64 = Block<'static, Vec<f64>>;

pub fn get_pooled_buffer(size: usize) -> PooledBuffer {
    let mut buf = BYTE_POOL.alloc(size);
    buf.clear();
    buf
}

static INT64_POOL: OnceLock<BytePool<Vec<i64>>> = OnceLock::new();

/// get_pooled_vec_i64 returns an int64 slice with the given size.
pub fn get_pooled_vec_i64(size: usize) -> PooledVecI64 {
    let pool = INT64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v.reserve(size);
    v
}

/// get_pool_vec_i64_filled returns an int64 slice with the given size and filled with the given value.
pub fn get_pool_vec_i64_filled(size: usize, value: i64) -> PooledVecI64 {
    let mut v = get_pooled_vec_i64(size);
    v.resize_with(size, || value);
    v
}

static F64_POOL: OnceLock<BytePool<Vec<f64>>> = OnceLock::new();

/// get_pooled_vec_f64 returns a f64 slice with the given size.
pub fn get_pooled_vec_f64(size: usize) -> PooledVecF64 {
    let pool = F64_POOL.get_or_init(BytePool::new);
    let mut v = pool.alloc(size);
    v.clear();
    v.reserve(size);
    v
}

/// get_pooled_vec_f64_filled returns a f64 slice with the given size and filled with the given value.
pub fn get_pooled_vec_f64_filled(size: usize, value: f64) -> PooledVecF64 {
    let mut v = get_pooled_vec_f64(size);
    v.resize_with(size, || value);
    v
}
