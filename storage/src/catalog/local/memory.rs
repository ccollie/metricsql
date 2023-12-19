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

use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use snafu::OptionExt;

use table::metadata::TableId;
use table::table::TableIdProvider;
use table::TableRef;

use crate::catalog::consts::{DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME, MIN_USER_TABLE_ID};
use crate::catalog::manager::{
    CatalogManager, DeregisterSchemaRequest, DeregisterTableRequest, RegisterSchemaRequest,
    RegisterTableRequest, RenameTableRequest,
};
use crate::error::{
    CatalogNotFoundSnafu, Result, SchemaNotFoundSnafu, TableExistsSnafu, TableNotFoundSnafu,
};
use crate::table::metadata::TableId;
use crate::table::{TableIdProvider, TableRef};
use crate::{
    CatalogManager, DeregisterSchemaRequest, DeregisterTableRequest, RegisterSchemaRequest,
    RegisterSystemTableRequest, RegisterTableRequest,
};

type SchemaEntries = HashMap<String, HashMap<String, TableRef>>;

/// Simple in-memory list of catalogs
pub struct MemoryCatalogManager {
    /// Collection of catalogs containing schemas and ultimately Tables
    pub catalogs: RwLock<HashMap<String, SchemaEntries>>,
    pub table_id: AtomicU32,
}

impl Default for MemoryCatalogManager {
    fn default() -> Self {
        let manager = Self {
            table_id: AtomicU32::new(MIN_USER_TABLE_ID),
            catalogs: Default::default(),
        };

        let catalog = HashMap::from([(DEFAULT_SCHEMA_NAME.to_string(), HashMap::new())]);
        let _ = manager
            .catalogs
            .write()
            .unwrap()
            .insert(DEFAULT_CATALOG_NAME.to_string(), catalog);

        manager
    }
}

#[async_trait::async_trait]
impl TableIdProvider for MemoryCatalogManager {
    async fn next_table_id(&self) -> table::error::Result<TableId> {
        Ok(self.table_id.fetch_add(1, Ordering::Relaxed))
    }
}

#[async_trait::async_trait]
impl CatalogManager for MemoryCatalogManager {
    async fn start(&self) -> Result<()> {
        self.table_id.store(MIN_USER_TABLE_ID, Ordering::Relaxed);
        Ok(())
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<bool> {
        self.register_table_sync(request)
    }

    async fn deregister_table(&self, request: DeregisterTableRequest) -> Result<()> {
        let mut catalogs = self.catalogs.write().unwrap();
        let schema = catalogs
            .get_mut(&request.catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: &request.catalog,
            })?
            .get_mut(&request.schema)
            .with_context(|| SchemaNotFoundSnafu {
                catalog: &request.catalog,
                schema: &request.schema,
            })?;
        let result = schema.remove(&request.table_name);
        Ok(())
    }

    async fn register_schema(&self, request: RegisterSchemaRequest) -> Result<bool> {
        self.register_schema_sync(request)
    }

    async fn deregister_schema(&self, request: DeregisterSchemaRequest) -> Result<bool> {
        let mut catalogs = self.catalogs.write().unwrap();
        let schemas = catalogs
            .get_mut(&request.catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: &request.catalog,
            })?;
        let table_count = schemas
            .remove(&request.schema)
            .with_context(|| SchemaNotFoundSnafu {
                catalog: &request.catalog,
                schema: &request.schema,
            })?
            .len();
        Ok(true)
    }

    async fn schema_exist(&self, catalog: &str, schema: &str) -> Result<bool> {
        Ok(self
            .catalogs
            .read()
            .unwrap()
            .get(catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: catalog,
            })?
            .contains_key(schema))
    }

    async fn table(
        &self,
        catalog: &str,
        schema: &str,
        table_name: &str,
    ) -> Result<Option<TableRef>> {
        let result = try {
            self.catalogs
                .read()
                .unwrap()
                .get(catalog)?
                .get(schema)?
                .get(table_name)
                .cloned()?
        };
        Ok(result)
    }

    async fn catalog_exist(&self, catalog: &str) -> Result<bool> {
        Ok(self.catalogs.read().unwrap().get(catalog).is_some())
    }

    async fn table_exist(&self, catalog: &str, schema: &str, table: &str) -> Result<bool> {
        let catalogs = self.catalogs.read().unwrap();
        Ok(catalogs
            .get(catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: catalog,
            })?
            .get(schema)
            .with_context(|| SchemaNotFoundSnafu { catalog, schema })?
            .contains_key(table))
    }

    async fn catalog_names(&self) -> Result<Vec<String>> {
        Ok(self.catalogs.read().unwrap().keys().cloned().collect())
    }

    async fn schema_names(&self, catalog_name: &str) -> Result<Vec<String>> {
        Ok(self
            .catalogs
            .read()
            .unwrap()
            .get(catalog_name)
            .with_context(|| CatalogNotFoundSnafu { catalog_name })?
            .keys()
            .cloned()
            .collect())
    }

    async fn table_names(&self, catalog_name: &str, schema_name: &str) -> Result<Vec<String>> {
        Ok(self
            .catalogs
            .read()
            .unwrap()
            .get(catalog_name)
            .with_context(|| CatalogNotFoundSnafu { catalog_name })?
            .get(schema_name)
            .with_context(|| SchemaNotFoundSnafu {
                catalog: catalog_name,
                schema: schema_name,
            })?
            .keys()
            .cloned()
            .collect())
    }

    async fn register_catalog(&self, name: String) -> Result<bool> {
        self.register_catalog_sync(name)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn rename_table(&self, request: RenameTableRequest) -> Result<bool> {
        todo!()
    }
}

impl MemoryCatalogManager {
    /// Registers a catalog and return the catalog already exist
    pub fn register_catalog_if_absent(&self, name: String) -> bool {
        let mut catalogs = self.catalogs.write().unwrap();
        let entry = catalogs.entry(name);
        match entry {
            Entry::Occupied(_) => true,
            Entry::Vacant(v) => {
                let _ = v.insert(HashMap::new());
                false
            }
        }
    }

    pub fn register_catalog_sync(&self, name: String) -> Result<bool> {
        let mut catalogs = self.catalogs.write().unwrap();

        match catalogs.entry(name) {
            Entry::Vacant(e) => {
                e.insert(HashMap::new());
                Ok(true)
            }
            Entry::Occupied(_) => Ok(false),
        }
    }

    pub fn register_schema_sync(&self, request: RegisterSchemaRequest) -> Result<bool> {
        let mut catalogs = self.catalogs.write().unwrap();
        let catalog = catalogs
            .get_mut(&request.catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: &request.catalog,
            })?;

        match catalog.entry(request.schema) {
            Entry::Vacant(e) => {
                e.insert(HashMap::new());
                Ok(true)
            }
            Entry::Occupied(_) => Ok(false),
        }
    }

    pub fn register_table_sync(&self, request: RegisterTableRequest) -> Result<bool> {
        let mut catalogs = self.catalogs.write().unwrap();
        let schema = catalogs
            .get_mut(&request.catalog)
            .with_context(|| CatalogNotFoundSnafu {
                catalog_name: &request.catalog,
            })?
            .get_mut(&request.schema)
            .with_context(|| SchemaNotFoundSnafu {
                catalog: &request.catalog,
                schema: &request.schema,
            })?;

        if schema.contains_key(&request.table_name) {
            return TableExistsSnafu {
                table: &request.table_name,
            }
            .fail();
        }
        schema.insert(request.table_name, request.table);
        Ok(true)
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn new_with_table(table: TableRef) -> Self {
        let manager = Self::default();
        let request = RegisterTableRequest {
            catalog: DEFAULT_CATALOG_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table_name: table.table_info().name.clone(),
            table_id: table.table_info().ident.table_id,
            table,
        };
        let _ = manager.register_table_sync(request).unwrap();
        manager
    }
}

/// Create a memory catalog list contains a numbers table for test
pub fn new_memory_catalog_manager() -> Result<Arc<MemoryCatalogManager>> {
    Ok(Arc::new(MemoryCatalogManager::default()))
}

#[cfg(test)]
mod tests {
    use common_catalog::consts::*;
    use common_error::ext::ErrorExt;
    use common_error::status_code::StatusCode;
    use table::table::numbers::{NumbersTable, NUMBERS_TABLE_NAME};

    use crate::catalog::consts::NUMBERS_TABLE_ID;

    use super::*;

    #[tokio::test]
    async fn test_new_memory_catalog_list() {
        let catalog_list = new_memory_catalog_manager().unwrap();

        let register_request = RegisterTableRequest {
            catalog: DEFAULT_CATALOG_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table_name: NUMBERS_TABLE_NAME.to_string(),
            table_id: NUMBERS_TABLE_ID,
            table: Arc::new(NumbersTable::default()),
        };

        let _ = catalog_list.register_table(register_request).await.unwrap();
        let table = catalog_list
            .table(
                DEFAULT_CATALOG_NAME,
                DEFAULT_SCHEMA_NAME,
                NUMBERS_TABLE_NAME,
            )
            .await
            .unwrap();
        let _ = table.unwrap();
        assert!(catalog_list
            .table(DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME, "not_exists")
            .await
            .unwrap()
            .is_none());
    }

    #[test]
    pub fn test_register_if_absent() {
        let list = MemoryCatalogManager::default();
        assert!(!list.register_catalog_if_absent("test_catalog".to_string()));
        assert!(list.register_catalog_if_absent("test_catalog".to_string()));
    }

    #[tokio::test]
    pub async fn test_catalog_deregister_table() {
        let catalog = MemoryCatalogManager::default();
        let table_name = "foo_table";

        let register_table_req = RegisterTableRequest {
            catalog: DEFAULT_CATALOG_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table_name: table_name.to_string(),
            table_id: 2333,
            table: Arc::new(NumbersTable::default()),
        };
        let _ = catalog.register_table(register_table_req).await.unwrap();
        assert!(catalog
            .table(DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME, table_name)
            .await
            .unwrap()
            .is_some());

        let deregister_table_req = DeregisterTableRequest {
            catalog: DEFAULT_CATALOG_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table_name: table_name.to_string(),
        };
        catalog
            .deregister_table(deregister_table_req)
            .await
            .unwrap();
        assert!(catalog
            .table(DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME, table_name)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn test_catalog_deregister_schema() {
        let catalog = MemoryCatalogManager::default();

        // Registers a catalog, a schema, and a table.
        let catalog_name = "foo_catalog".to_string();
        let schema_name = "foo_schema".to_string();
        let table_name = "foo_table".to_string();
        let schema = RegisterSchemaRequest {
            catalog: catalog_name.clone(),
            schema: schema_name.clone(),
        };
        let table = RegisterTableRequest {
            catalog: catalog_name.clone(),
            schema: schema_name.clone(),
            table_name,
            table_id: 0,
            table: Arc::new(NumbersTable::default()),
        };
        catalog
            .register_catalog(catalog_name.clone())
            .await
            .unwrap();
        catalog.register_schema(schema).await.unwrap();
        catalog.register_table(table).await.unwrap();

        let request = DeregisterSchemaRequest {
            catalog: catalog_name.clone(),
            schema: schema_name.clone(),
        };

        assert!(catalog.deregister_schema(request).await.unwrap());
        assert!(!catalog
            .schema_exist(&catalog_name, &schema_name)
            .await
            .unwrap());
    }
}
