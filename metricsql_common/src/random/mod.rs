
pub fn random() -> u8 {
    rand::random()
}

pub fn rand_next32() -> u32 {
    rand::random::<u32>()
}

pub fn rand_next64() -> u64 {
    rand::random::<u64>()
}