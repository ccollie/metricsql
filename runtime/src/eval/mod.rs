pub(crate) use binary::merge_non_overlapping_timeseries;
pub use dag::{compile_expression, DAGNode};
pub use eval::*;
pub use traits::*;

pub mod binary;
mod dag;
mod eval;
#[cfg(test)]
mod eval_test;
mod traits;
mod utils;
