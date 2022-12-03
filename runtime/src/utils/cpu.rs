use crate::runtime_error::{RuntimeError, RuntimeResult};
use std::num::NonZeroUsize;

pub fn num_cpus() -> RuntimeResult<NonZeroUsize> {
    match std::thread::available_parallelism() {
        Err(_) => Err(RuntimeError::General(
            "Error fetching available_parallelism".to_string(),
        )),
        Ok(v) => Ok(v),
    }
}
