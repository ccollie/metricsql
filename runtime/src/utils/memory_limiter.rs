// todo: have config flag for wasm
use std::sync::{Arc, Mutex};
use crate::runtime_error::{RuntimeError, RuntimeResult};

struct Inner {
    usage: usize
}

#[derive(Default)]
pub struct MemoryLimiter {
    // todo; use AtomicUsize
    inner: Arc<Mutex<usize>>,
    pub max_size: usize
}

impl MemoryLimiter {
    pub fn new(max_size: usize) -> Self {
        MemoryLimiter {
            inner: Arc::new(Mutex::new(0)),
            max_size,
        }
    }

    pub fn get(&mut self, n: usize) -> bool {
        // read() will only block when `producer_thread` is holding a write lock
        if let Ok(mut usage) = self.inner.lock() {
            if n <= self.max_size && self.max_size-n >= *usage {
                *usage += n;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn put(&mut self, n: usize) -> RuntimeResult<()> {
        let mut inner = self.inner.lock().unwrap();
        if n > *inner {
            return Err(RuntimeError::from(format!("MemoryLimiter: n={} cannot exceed {}", n, *inner)));
        }
        *inner -= n;
        Ok(())
    }
}
