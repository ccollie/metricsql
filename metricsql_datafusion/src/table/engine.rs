// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::table::path_utils::DATA_DIR;
use std::fmt::{self, Display};
use std::sync::Arc;

use crate::error::Result;
use crate::table::error::UnsupportedSnafu;
use crate::table::metadata::TableId;
use crate::table::requests::{
    CloseTableRequest, CreateTableRequest, DropTableRequest, OpenTableRequest,
};
use crate::table::table::TableRef;

mod manager;

/// Represents a resolved path to a table of the form “catalog.schema.table”
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableReference<'a> {
    pub catalog: &'a str,
    pub schema: &'a str,
    pub table: &'a str,
}

// TODO(LFC): Find a better place for `TableReference`,
// so that we can reuse the default catalog and schema consts.
// Could be done together with issue #559.
impl<'a> TableReference<'a> {
    pub fn bare(table: &'a str) -> Self {
        TableReference {
            catalog: "metrix",
            schema: "public",
            table,
        }
    }

    pub fn full(catalog: &'a str, schema: &'a str, table: &'a str) -> Self {
        TableReference {
            catalog,
            schema,
            table,
        }
    }
}

impl<'a> Display for TableReference<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.catalog, self.schema, self.table)
    }
}

/// CloseTableResult
///
/// Returns [`CloseTableResult::Released`] and closed region numbers if a table was removed
/// from the query_engine.
#[derive(Debug)]
pub enum CloseTableResult {
    Released,
    NotFound,
}

/// Table query_engine abstraction.
#[async_trait::async_trait]
pub trait TableEngine: Send + Sync {
    /// Return query_engine name
    fn name(&self) -> &str;

    /// Create a table by given request.
    ///
    /// Return the created table.
    async fn create_table(
        &self,
        ctx: &EngineContext,
        request: CreateTableRequest,
    ) -> Result<TableRef>;

    /// Open an existing table by given `request`, returns the opened table. If the table does not
    /// exist, returns an `Ok(None)`.
    async fn open_table(
        &self,
        ctx: &EngineContext,
        request: OpenTableRequest,
    ) -> Result<Option<TableRef>>;

    /// Returns the table by it's name.
    fn get_table(&self, ctx: &EngineContext, table_id: TableId) -> Result<Option<TableRef>>;

    /// Returns true when the given table is exists.
    fn table_exists(&self, ctx: &EngineContext, table_id: TableId) -> bool;

    /// Drops the given table. Return true if the table is dropped, or false if the table doesn't exist.
    async fn drop_table(&self, ctx: &EngineContext, request: DropTableRequest) -> Result<bool>;

    /// Closes the (partial) given table.
    ///
    /// Removes a table from the query_engine if all regions are closed.
    async fn close_table(
        &self,
        _ctx: &EngineContext,
        _request: CloseTableRequest,
    ) -> Result<CloseTableResult> {
        UnsupportedSnafu {
            operation: "close_table",
        }
        .fail()?
    }

    /// Close the query_engine.
    async fn close(&self) -> Result<()>;
}

pub type TableEngineRef = Arc<dyn TableEngine>;

/// Table query_engine context.
#[derive(Debug, Clone, Default)]
pub struct EngineContext {}

#[inline]
pub fn table_dir(catalog_name: &str, schema_name: &str, table_id: TableId) -> String {
    //let dir = DATA_DIR;
    format!("{DATA_DIR}{catalog_name}/{schema_name}/{table_id}/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_reference() {
        let table_ref = TableReference {
            catalog: "metrix",
            schema: "public",
            table: "test",
        };

        assert_eq!("metrix.public.test", table_ref.to_string());
    }
}
