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

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use datafusion_expr::TableType;
use futures_util::StreamExt;
use itertools::Itertools;
use lazy_static::lazy_static;
use paste::paste;
use snafu::ResultExt;

use metricsql_common::error::ext::BoxedError;
pub use table_names::*;

use crate::catalog::CatalogManager;
use crate::catalog::consts::INFORMATION_SCHEMA_NAME;
use crate::catalog::information_schema::memory_table::{get_schema_columns, MemoryTable};
use crate::catalog::information_schema::tables::InformationSchemaTables;
use crate::common::recordbatch::{RecordBatchStreamWrapper, SendableRecordBatchStream};
use crate::data_source::DataSource;
use crate::table::{TableId, TableRef};
use crate::table::error::{SchemaConversionSnafu, TablesRecordBatchSnafu};
use crate::table::metadata::{
    FilterPushDownType, TableInfoBuilder, TableInfoRef, TableMetaBuilder,
};
use crate::table::storage::ScanRequest;
use crate::table::thin_table::{ThinTable, ThinTableAdapter};

use super::error::Result;

use self::columns::InformationSchemaColumns;

mod columns;
mod memory_table;
mod table_names;
mod tables;

lazy_static! {
    // Memory tables in `information_schema`.
    static ref MEMORY_TABLES: &'static [&'static str] = &[
        ENGINES,
        COLUMN_PRIVILEGES,
        COLUMN_STATISTICS,
        BUILD_INFO,
    ];
}

macro_rules! setup_memory_table {
    ($name: expr) => {
        paste! {
            {
                let (schema, columns) = get_schema_columns($name);
                Some(Arc::new(MemoryTable::new(
                    crate::catalog::consts::[<INFORMATION_SCHEMA_ $name  _TABLE_ID>],
                    $name,
                    schema,
                    columns
                )) as _)
            }
        }
    };
}

/// The `information_schema` tables info provider.
pub struct InformationSchemaProvider {
    catalog_name: String,
    catalog_manager: Weak<dyn CatalogManager>,
    tables: HashMap<String, TableRef>,
}

impl InformationSchemaProvider {
    pub fn new(catalog_name: String, catalog_manager: Weak<dyn CatalogManager>) -> Self {
        let mut provider = Self {
            catalog_name,
            catalog_manager,
            tables: HashMap::new(),
        };

        provider.build_tables();

        provider
    }

    /// Returns table names.
    pub fn table_names(&self) -> Vec<String> {
        let mut tables = self.tables.values().clone().collect::<Vec<_>>();

        tables
            .into_iter()
            .map(|t| t.table_info().name.clone())
            .sorted_by(|a, b| a.cmp(b))
            .collect()
    }

    /// Returns a map of [TableRef] in information schema.
    pub fn tables(&self) -> &HashMap<String, TableRef> {
        assert!(!self.tables.is_empty());

        &self.tables
    }

    /// Returns the [TableRef] by table name.
    pub fn table(&self, name: &str) -> Option<TableRef> {
        self.tables.get(name).cloned()
    }

    fn build_tables(&mut self) {
        let mut tables = HashMap::new();
        tables.insert(TABLES.to_string(), self.build_table(TABLES).unwrap());
        tables.insert(COLUMNS.to_string(), self.build_table(COLUMNS).unwrap());

        // Add memory tables
        for name in MEMORY_TABLES.iter() {
            tables.insert((*name).to_string(), self.build_table(name).unwrap());
        }

        self.tables = tables;
    }

    fn build_table(&self, name: &str) -> Option<TableRef> {
        self.information_table(name).map(|table| {
            let table_info = Self::table_info(self.catalog_name.clone(), &table);
            let filter_pushdown = FilterPushDownType::Unsupported;
            let thin_table = ThinTable::new(table_info, filter_pushdown);

            let data_source = Arc::new(InformationTableDataSource::new(table));
            Arc::new(ThinTableAdapter::new(thin_table, data_source)) as _
        })
    }

    fn information_table(&self, name: &str) -> Option<InformationTableRef> {
        match name.to_ascii_lowercase().as_str() {
            TABLES => Some(Arc::new(InformationSchemaTables::new(
                self.catalog_name.clone(),
                self.catalog_manager.clone(),
            )) as _),
            COLUMNS => Some(Arc::new(InformationSchemaColumns::new(
                self.catalog_name.clone(),
                self.catalog_manager.clone(),
            )) as _),
            ENGINES => setup_memory_table!(ENGINES),
            COLUMN_PRIVILEGES => setup_memory_table!(COLUMN_PRIVILEGES),
            COLUMN_STATISTICS => setup_memory_table!(COLUMN_STATISTICS),
            BUILD_INFO => setup_memory_table!(BUILD_INFO),
            _ => None,
        }
    }

    fn table_info(catalog_name: String, table: &InformationTableRef) -> TableInfoRef {
        let table_meta = TableMetaBuilder::default()
            .schema(table.schema())
            .primary_key_indices(vec![])
            .build()
            .unwrap();
        let table_info = TableInfoBuilder::default()
            .name(table.table_name().to_string())
            .catalog_name(catalog_name)
            .schema_name(INFORMATION_SCHEMA_NAME.to_string())
            .meta(table_meta)
            .table_type(table.table_type())
            .build()
            .unwrap();
        Arc::new(table_info)
    }
}

trait InformationTable {
    fn table_id(&self) -> TableId;

    fn table_name(&self) -> &'static str;

    fn schema(&self) -> SchemaRef;

    fn to_stream(&self) -> Result<SendableRecordBatchStream>;

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }
}

type InformationTableRef = Arc<dyn InformationTable + Send + Sync>;

struct InformationTableDataSource {
    table: InformationTableRef,
}

impl InformationTableDataSource {
    fn new(table: InformationTableRef) -> Self {
        Self { table }
    }

    fn try_project(&self, projection: &[usize]) -> std::result::Result<SchemaRef, BoxedError> {
        let schema = self
            .table
            .schema()
            .project(projection)
            .context(SchemaConversionSnafu)
            .map_err(BoxedError::new)?;
        Ok(Arc::new(schema))
    }
}

impl DataSource for InformationTableDataSource {
    fn get_stream(
        &self,
        request: ScanRequest,
    ) -> std::result::Result<SendableRecordBatchStream, BoxedError> {
        let projection = request.projection;
        let projected_schema = match &projection {
            Some(projection) => self.try_project(projection)?,
            None => self.table.schema(),
        };

        let stream = self
            .table
            .to_stream()
            .map_err(BoxedError::new)
            .context(TablesRecordBatchSnafu)
            .map_err(BoxedError::new)?
            .map(move |batch| match &projection {
                Some(p) => batch.and_then(|b| b.try_project(p)),
                None => batch,
            });

        let stream = RecordBatchStreamWrapper {
            schema: projected_schema,
            stream: Box::pin(stream),
            output_ordering: None,
        };

        Ok(Box::pin(stream))
    }
}
