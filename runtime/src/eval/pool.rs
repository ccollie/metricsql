use byte_pool::{Block, BytePool};
use once_cell::sync::Lazy;

static BYTE_POOL: Lazy<BytePool<Vec<i64>>> = Lazy::new(BytePool::<Vec<i64>>::new);

pub fn get_timestamp_buffer(size: usize) -> Block<'static, Vec<i64>> {
    let mut buf = BYTE_POOL.alloc(size);
    buf.clear();
    buf
}
