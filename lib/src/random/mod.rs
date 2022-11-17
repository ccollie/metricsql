use rand::{Rng, thread_rng};
use rand_distr::StandardNormal;

pub fn random() -> u8 {
    rand::random()
}

pub fn rand_next32() -> u32 {
    rand::random::<u32>()
}

pub fn rand_next64() -> u64 {
    rand::random::<u64>()
}

pub fn rand_nextf64() -> f64 {
    thread_rng().sample(StandardNormal)
}