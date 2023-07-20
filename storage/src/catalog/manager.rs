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

#![feature(trait_upcasting)]
#![feature(assert_matches)]

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use snafu::ResultExt;
use tracing::info;

use api::v1::meta::{RegionStat, TableIdent, TableName};
use common_telemetry::{info, warn};
use table::engine::{EngineContext, TableEngineRef};
use table::metadata::TableId;
use table::requests::CreateTableRequest;
use table::TableRef;

use crate::error::{CreateTableSnafu, Result};

#[async_trait::async_trait]
pub trait CatalogManager: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Starts a catalog manager.
    async fn start(&self) -> Result<()>;

    /// Registers a catalog to catalog manager, returns whether the catalog exist before.
    async fn register_catalog(&self, name: String) -> Result<bool>;

    /// Register a schema with catalog name and schema name. Returns whether the
    /// schema registered.
    ///
    /// # Errors
    ///
    /// This method will/should fail if catalog not exist
    async fn register_schema(&self, request: RegisterSchemaRequest) -> Result<bool>;

    /// Deregisters a database within given catalog/schema to catalog manager
    async fn deregister_schema(&self, request: DeregisterSchemaRequest) -> Result<bool>;

    /// Registers a table within given catalog/schema to catalog manager,
    /// returns whether the table registered.
    ///
    /// # Errors
    ///
    /// This method will/should fail if catalog or schema not exist
    async fn register_table(&self, request: RegisterTableRequest) -> Result<bool>;

    /// Deregisters a table within given catalog/schema to catalog manager
    async fn deregister_table(&self, request: DeregisterTableRequest) -> Result<()>;

    /// Rename a table to [RenameTableRequest::new_table_name], returns whether the table is renamed.
    async fn rename_table(&self, request: RenameTableRequest) -> Result<bool>;

    /// Register a system table, should be called before starting the manager.
    async fn register_system_table(&self, request: RegisterSystemTableRequest)
        -> error::Result<()>;

    async fn catalog_names(&self) -> Result<Vec<String>>;

    async fn schema_names(&self, catalog: &str) -> Result<Vec<String>>;

    async fn table_names(&self, catalog: &str, schema: &str) -> Result<Vec<String>>;

    async fn catalog_exist(&self, catalog: &str) -> Result<bool>;

    async fn schema_exist(&self, catalog: &str, schema: &str) -> Result<bool>;

    async fn table_exist(&self, catalog: &str, schema: &str, table: &str) -> Result<bool>;

    /// Returns the table by catalog, schema and table name.
    async fn table(
        &self,
        catalog: &str,
        schema: &str,
        table_name: &str,
    ) -> Result<Option<TableRef>>;
}

pub type CatalogManagerRef = Arc<dyn CatalogManager>;

/// Hook called after system table opening.
pub type OpenSystemTableHook = Arc<dyn Fn(TableRef) -> Result<()> + Send + Sync>;

/// Register system table request:
/// - When system table is already created and registered, the hook will be called
///     with table ref after opening the system table
/// - When system table does not exist, create and register the table by create_table_request and calls open_hook with the created table.
pub struct RegisterSystemTableRequest {
    pub create_table_request: CreateTableRequest,
    pub open_hook: Option<OpenSystemTableHook>,
}

#[derive(Clone)]
pub struct RegisterTableRequest {
    pub catalog: String,
    pub schema: String,
    pub table_name: String,
    pub table_id: TableId,
    pub table: TableRef,
}

impl Debug for RegisterTableRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterTableRequest")
            .field("catalog", &self.catalog)
            .field("schema", &self.schema)
            .field("table_name", &self.table_name)
            .field("table_id", &self.table_id)
            .field("table", &self.table.table_info())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct CreateTableRequest {
    pub catalog_name: String,
    pub schema_name: String,
    pub table_name: String,
    pub id: TableId,
    pub engine: String,
    pub create_if_not_exists: bool,
    pub desc: Option<String>,
    pub schema: RawSchema,
    pub primary_key_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RenameTableRequest {
    pub catalog: String,
    pub schema: String,
    pub table_name: String,
    pub new_table_name: String,
    pub table_id: TableId,
}

#[derive(Debug, Clone)]
pub struct DeregisterTableRequest {
    pub catalog: String,
    pub schema: String,
    pub table_name: String,
}

#[derive(Debug, Clone)]
pub struct DeregisterSchemaRequest {
    pub catalog: String,
    pub schema: String,
}

#[derive(Debug, Clone)]
pub struct RegisterSchemaRequest {
    pub catalog: String,
    pub schema: String,
}

pub(crate) async fn handle_system_table_request<'a, M: CatalogManager>(
    manager: &'a M,
    engine: TableEngineRef,
    sys_table_requests: &'a mut Vec<RegisterSystemTableRequest>,
) -> Result<()> {
    for req in sys_table_requests.drain(..) {
        let catalog_name = &req.create_table_request.catalog_name;
        let schema_name = &req.create_table_request.schema_name;
        let table_name = &req.create_table_request.table_name;
        let table_id = req.create_table_request.id;

        let table = manager.table(catalog_name, schema_name, table_name).await?;
        let table = if let Some(table) = table {
            table
        } else {
            let table = engine
                .create_table(&EngineContext::default(), req.create_table_request.clone())
                .await
                .with_context(|_| CreateTableSnafu {
                    table_info: common_catalog::format_full_table_name(
                        catalog_name,
                        schema_name,
                        table_name,
                    ),
                })?;
            let _ = manager
                .register_table(RegisterTableRequest {
                    catalog: catalog_name.clone(),
                    schema: schema_name.clone(),
                    table_name: table_name.clone(),
                    table_id,
                    table: table.clone(),
                })
                .await?;
            info!("Created and registered system table: {table_name}");
            table
        };
        if let Some(hook) = req.open_hook {
            (hook)(table)?;
        }
    }
    Ok(())
}