use byte_pool::{Block, BytePool};
use once_cell::sync::Lazy;

static BYTE_POOL: Lazy<BytePool<Vec<u8>>> = Lazy::new(BytePool::<Vec<u8>>::new);

pub fn get_pooled_buffer(size: usize) -> Block<'static, Vec<u8>> {
    let mut buf = BYTE_POOL.alloc(size);
    buf.clear();
    buf
}
