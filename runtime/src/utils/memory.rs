use crate::runtime_error::{RuntimeError, RuntimeResult};

// todo: config for wasm.
pub fn memory_limit() -> RuntimeResult<u64> {
    match effective_limits::memory_limit() {
        Ok(v) => Ok(v),
        Err(_) => {
            Err(RuntimeError::General("Error getting effective_memory limits".to_string()))
        }
    }
}