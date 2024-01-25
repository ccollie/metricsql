#![feature(assert_matches)]
pub(crate) mod planner;
pub mod error;
mod common;
pub mod table;
pub mod catalog;
pub mod datatypes;
pub mod data_source;
pub(crate) mod datasource;
pub(crate) mod file_engine;
#[cfg(test)]
pub(crate) mod test_util;
mod object_store;
mod query;
mod sql;
mod udf;
mod session;
mod provider;

pub mod prelude {
    pub use crate::provider::*;
}
