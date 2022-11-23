
pub fn num_cpus() {
    std::thread::available_parallelism()
}