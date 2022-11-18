mod compress;
mod encoding;
mod int;
mod nearest_delta;
mod nearest_delta2;
mod float;


pub use compress::*;
pub use encoding::*;
pub use int::*;
pub use float::*;
pub use nearest_delta::*;
pub use nearest_delta2::*;

// todo: move to sep file ?
#[cfg(test)]
mod int_test;
#[cfg(test)]
mod nearest_delta2_test;
#[cfg(test)]
mod encoding_pure_test;
#[cfg(test)]
mod encoding_test;
#[cfg(test)]
mod nearest_delta_test;
