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

use std::sync::{Arc, Weak};

use arrow::array::ArrayRef;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter as DfRecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream as DfPartitionStream;
use datafusion::physical_plan::SendableRecordBatchStream as DfSendableRecordBatchStream;
use datafusion_expr::TableType;
use snafu::{OptionExt, ResultExt};

use metricsql_common::prelude::BoxedError;

use crate::catalog::consts::INFORMATION_SCHEMA_TABLES_TABLE_ID;
use crate::catalog::error::{CreateRecordBatchSnafu, InternalSnafu, Result, UpgradeWeakCatalogManagerRefSnafu};
use crate::catalog::CatalogManager;
use crate::catalog::{StringVectorBuilder, UInt32VectorBuilder};
use crate::common::recordbatch::adapter::RecordBatchStreamAdapter;
use crate::common::recordbatch::{RecordBatch, SendableRecordBatchStream};
use crate::table::TableId;

use super::{InformationTable, TABLES};

pub(super) struct InformationSchemaTables {
    schema: SchemaRef,
    catalog_name: String,
    catalog_manager: Weak<dyn CatalogManager>,
}

impl InformationSchemaTables {
    pub(super) fn new(catalog_name: String, catalog_manager: Weak<dyn CatalogManager>) -> Self {
        Self {
            schema: Self::schema(),
            catalog_name,
            catalog_manager,
        }
    }

    pub(crate) fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("table_catalog", DataType::Utf8, false),
            Field::new("table_schema", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("table_type", DataType::Utf8, false),
            Field::new("table_id", DataType::UInt32, true),
            Field::new("query_engine", DataType::Utf8, true),
        ]))
    }

    fn builder(&self) -> InformationSchemaTablesBuilder {
        InformationSchemaTablesBuilder::new(
            self.schema.clone(),
            self.catalog_name.clone(),
            self.catalog_manager.clone(),
        )
    }
}

impl InformationTable for InformationSchemaTables {
    fn table_id(&self) -> TableId {
        INFORMATION_SCHEMA_TABLES_TABLE_ID
    }

    fn table_name(&self) -> &'static str {
        TABLES
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn to_stream(&self) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.clone();
        let mut builder = self.builder();
        let stream = Box::pin(DfRecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move {
                builder
                    .make_tables()
                    .await
                    .map(|x| x.into_df_record_batch())
                    .map_err(Into::into)
            }),
        ));
        Ok(Box::pin(
            RecordBatchStreamAdapter::try_new(stream)
                .map_err(BoxedError::new)
                .context(InternalSnafu)?,
        ))
    }
}

/// Builds the `information_schema.TABLE` table row by row
///
/// Columns are based on <https://www.postgresql.org/docs/current/infoschema-columns.html>
struct InformationSchemaTablesBuilder {
    schema: SchemaRef,
    catalog_name: String,
    catalog_manager: Weak<dyn CatalogManager>,

    catalog_names: StringVectorBuilder,
    schema_names: StringVectorBuilder,
    table_names: StringVectorBuilder,
    table_types: StringVectorBuilder,
    table_ids: UInt32VectorBuilder,
    engines: StringVectorBuilder,
}

impl InformationSchemaTablesBuilder {
    fn new(
        schema: SchemaRef,
        catalog_name: String,
        catalog_manager: Weak<dyn CatalogManager>,
    ) -> Self {
        Self {
            schema,
            catalog_name,
            catalog_manager,
            catalog_names: StringVectorBuilder::with_capacity(42),
            schema_names: StringVectorBuilder::with_capacity(42),
            table_names: StringVectorBuilder::with_capacity(42),
            table_types: StringVectorBuilder::with_capacity(42),
            table_ids: UInt32VectorBuilder::with_capacity(42),
            engines: StringVectorBuilder::with_capacity(42),
        }
    }

    /// Construct the `information_schema.tables` virtual table
    async fn make_tables(&mut self) -> Result<RecordBatch> {
        let catalog_name = self.catalog_name.clone();
        let catalog_manager = self
            .catalog_manager
            .upgrade()
            .context(UpgradeWeakCatalogManagerRefSnafu)?;

        for schema_name in catalog_manager.schema_names(&catalog_name).await? {
            if !catalog_manager
                .schema_exists(&catalog_name, &schema_name)
                .await?
            {
                continue;
            }

            for table_name in catalog_manager
                .table_names(&catalog_name, &schema_name)
                .await?
            {
                if let Some(table) = catalog_manager
                    .table(&catalog_name, &schema_name, &table_name)
                    .await?
                {
                    let table_info = table.table_info();
                    self.add_table(
                        &catalog_name,
                        &schema_name,
                        &table_name,
                        table.table_type(),
                        Some(&table_info.meta.engine),
                    );
                } else {
                    unreachable!();
                }
            }
        }

        self.finish()
    }

    fn add_table(
        &mut self,
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
        table_type: TableType,
        engine: Option<&str>,
    ) {
        self.catalog_names.push(Some(catalog_name));
        self.schema_names.push(Some(schema_name));
        self.table_names.push(Some(table_name));
        self.table_types.push(Some(match table_type {
            TableType::Base => "BASE TABLE",
            TableType::View => "VIEW",
            TableType::Temporary => "LOCAL TEMPORARY",
        }));
        self.engines.push(engine);
    }

    fn finish(&mut self) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.catalog_names.finish()),
            Arc::new(self.schema_names.finish()),
            Arc::new(self.table_names.finish()),
            Arc::new(self.table_types.finish()),
            Arc::new(self.engines.finish()),
        ];
        RecordBatch::new(self.schema.clone(), columns).context(CreateRecordBatchSnafu)
    }
}

impl DfPartitionStream for InformationSchemaTables {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _: Arc<TaskContext>) -> DfSendableRecordBatchStream {
        let schema = self.schema.clone();
        let mut builder = self.builder();
        Box::pin(DfRecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move {
                builder
                    .make_tables()
                    .await
                    .map(|x| x.into_df_record_batch())
                    .map_err(Into::into)
            }),
        ))
    }
}
