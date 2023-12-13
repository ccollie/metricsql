use ::rand::prelude::*;

pub use rand::*;

mod generators;
mod mackey_glass;
mod rand;

pub fn create_rng(seed: Option<u64>) -> Result<StdRng, String> {
    if let Some(seed) = seed {
        Ok(StdRng::seed_from_u64(seed))
    } else {
        match StdRng::from_rng(thread_rng()) {
            Err(e) => Err(format!("Error constructing rng {:?}", e).to_string()),
            Ok(rng) => Ok(rng),
        }
    }
}
