// todo: have config flag for wasm

struct Inner {
    max_size: usize,
    usage: usize,
    in_fligbt: Vec<usize>
}

#[derive(Default)]
pub(crate) struct MemoryLimiter {
    inner: Arc<RwLock<Inner>>
}

impl MemoryLimiter {
    pub fn new(max_size: usize) -> Self {
        let inner = Inner {
            max_size,
            usage: 0,
            in_flight: Vec::with_capacity(1)
        };
        MemoryLimiter {
            inner: Arc::new(RwLock::new(inner))
        }
    }

    pub fn get(&mut self, n: usize) -> bool {
        // read() will only block when `producer_thread` is holding a write lock
        if let Ok(data) = self.inner.read() {
            let ok = n <= data.max_size && data.max_size-n >= data.usage;
            if ok {
                inner.usage += n;
                data.in_flight.push(n);
            }
            return ok
        } else {
            false
        }
    }

    pub fn put(&mut self, n: usize) -> RuntimeResult<()> {
        let inner = self.inner.write().unwrap();
        if n > inner.ml.usage {
            return Err(RuntimeError::from(format!("BUG: n=%d cannot exceed %d", n, ml.usage)));
        }
        *inner.ml.usage -= n;
        Ok(())
    }
}
