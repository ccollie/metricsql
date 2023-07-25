extern crate anyhow;
extern crate core;
extern crate datafusion;
#[macro_use]
extern crate derive_builder;
extern crate once_cell;
extern crate regex;
extern crate regex_syntax;
extern crate snafu;
extern crate tokio;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::Schema;
use datafusion::error::DataFusionError;
use datafusion::prelude::SessionContext;

mod catalog;
mod common;
mod error;
mod status_code;
mod table;
mod udf;
mod utils;
mod r#object

pub struct TableContext {
    pub session: SessionContext,
    pub name: String,
    pub schema: Arc<Schema>,
    pub timestamp_column: String,
    pub value_column: String,
}

#[async_trait]
pub trait TableProvider: Sync + Send + 'static {
    async fn create_context(
        &self,
        name: &str,
        time_range: (i64, i64),
        filters: &[(&str, &str)],
    ) -> Result<Vec<TableContext>, DataFusionError>;
}
