// todo: have config flag for wasm
use crate::runtime_error::{RuntimeError, RuntimeResult};
use std::sync::Mutex;

#[derive(Default)]
pub struct MemoryLimiter {
    // todo; use AtomicUsize
    inner: Mutex<usize>,
    pub max_size: usize,
}

impl MemoryLimiter {
    pub fn new(max_size: usize) -> Self {
        MemoryLimiter {
            inner: Mutex::new(0),
            max_size,
        }
    }

    pub fn get(&self, n: usize) -> bool {
        // read() will only block when `producer_thread` is holding a write lock
        if let Ok(mut usage) = self.inner.lock() {
            if n <= self.max_size && self.max_size - n >= *usage {
                *usage += n;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn put(&self, n: usize) -> RuntimeResult<()> {
        let mut inner = self.inner.lock().unwrap();
        if n > *inner {
            // todo: better error enum
            return Err(RuntimeError::from(format!(
                "MemoryLimiter: n={} cannot exceed {}",
                n, *inner
            )));
        }
        *inner -= n;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn usage(&self) -> usize {
        *self.inner.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::common::memory_limiter::MemoryLimiter;

    #[test]
    fn test_memory_limiter() {
        let ml = MemoryLimiter::new(100);

        // Allocate memory
        assert!(ml.get(10), "cannot get 10 out of {} bytes", ml.max_size);

        assert_eq!(
            ml.usage(),
            10,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            10
        );
        assert!(ml.get(20), "cannot get 20 out of 90 bytes");

        assert_eq!(
            ml.usage(),
            30,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            30
        );
        assert_ne!(ml.get(1000), true, "unexpected get for 1000 bytes");
        assert_eq!(
            ml.usage(),
            30,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            30
        );
        assert_ne!(ml.get(71), true, "unexpected get for 71 bytes");

        assert_eq!(
            ml.usage(),
            30,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            30
        );
        assert!(ml.get(70), "cannot get 70 bytes");

        assert_eq!(
            ml.usage(),
            100,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            100
        );

        // Return memory back
        ml.put(10).expect("error returning memory");
        ml.put(70).expect("error returning memory");
        assert_eq!(
            ml.usage(),
            20,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            20
        );
        assert!(ml.get(30), "cannot get 30 bytes");
        ml.put(50).expect("error returning memory");
        assert_eq!(
            ml.usage(),
            0,
            "unexpected usage; got {}; want {}",
            ml.usage(),
            0
        );
    }
}
