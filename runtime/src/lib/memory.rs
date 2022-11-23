// todo: config for wasm.
pub fn memory_limit() -> Result<u64, Error> {
    effective_limits::memory_limit()
}