use std::num::NonZero;
use std::sync::LazyLock;
use std::num::NonZeroUsize;

static NUM_CPUS: LazyLock<NonZeroUsize> = LazyLock::new(|| {
    // todo: log info on error
    std::thread::available_parallelism().unwrap_or(NonZero::new(1usize).expect("BUG: NonZero(1) panic"))
});

pub fn num_cpus() -> NonZeroUsize {
    *NUM_CPUS
}
