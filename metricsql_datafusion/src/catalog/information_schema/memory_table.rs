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

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow::array::ArrayRef;
use arrow_schema::{SchemaRef as ArrowSchemaRef, SchemaRef};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::SendableRecordBatchStream as DfSendableRecordBatchStream;
use datafusion::physical_plan::stream::{RecordBatchStreamAdapter as DfRecordBatchStreamAdapter, RecordBatchStreamAdapter};
use datafusion::physical_plan::streaming::PartitionStream as DfPartitionStream;
use snafu::ResultExt;

use metricsql_common::prelude::BoxedError;
pub use tables::get_schema_columns;

use crate::catalog::error::{CreateRecordBatchSnafu, Result};
use crate::catalog::error::InternalSnafu;
use crate::catalog::information_schema::InformationTable;
use crate::table::TableId;

mod tables;

/// A memory table with specified schema and columns.
pub(super) struct MemoryTable {
    table_id: TableId,
    table_name: &'static str,
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
}

impl MemoryTable {
    /// Creates a memory table with table id, name, schema and columns.
    pub(super) fn new(
        table_id: TableId,
        table_name: &'static str,
        schema: SchemaRef,
        columns: Vec<ArrayRef>,
    ) -> Self {
        Self {
            table_id,
            table_name,
            schema,
            columns,
        }
    }

    fn builder(&self) -> MemoryTableBuilder {
        MemoryTableBuilder::new(self.schema.clone(), self.columns.clone())
    }
}

impl InformationTable for MemoryTable {
    fn table_id(&self) -> TableId {
        self.table_id
    }

    fn table_name(&self) -> &'static str {
        self.table_name
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn to_stream(&self) -> Result<SendableRecordBatchStream> {
        let schema = self.schema.arrow_schema().clone();
        let mut builder = self.builder();
        let stream = Box::pin(DfRecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move {
                builder
                    .memory_records()
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

struct MemoryTableBuilder {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
}

impl MemoryTableBuilder {
    fn new(schema: SchemaRef, columns: Vec<ArrayRef>) -> Self {
        Self { schema, columns }
    }

    /// Construct the `information_schema.{table_name}` virtual table
    async fn memory_records(&mut self) -> Result<RecordBatch> {
        if self.columns.is_empty() {
            RecordBatch::new_empty(self.schema.clone()).context(CreateRecordBatchSnafu)
        } else {
            RecordBatch::new(self.schema.clone(), std::mem::take(&mut self.columns))
                .context(CreateRecordBatchSnafu)
        }
    }
}

impl DfPartitionStream for MemoryTable {
    fn schema(&self) -> &ArrowSchemaRef {
        self.schema.arrow_schema()
    }

    fn execute(&self, _: Arc<TaskContext>) -> DfSendableRecordBatchStream {
        let schema = self.schema.arrow_schema().clone();
        let mut builder = self.builder();
        Box::pin(DfRecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move {
                builder
                    .memory_records()
                    .await
                    .map(|x| x.into_df_record_batch())
                    .map_err(Into::into)
            }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};

    use crate::common::recordbatch::RecordBatches;
    use crate::table::schema::column_schema::Field;

    use super::*;

    #[tokio::test]
    async fn test_memory_table() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]));

        let table = MemoryTable::new(
            42,
            "test",
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a1", "a2"])),
                Arc::new(StringArray::from(vec!["b1", "b2"])),
            ],
        );

        assert_eq!(42, table.table_id());
        assert_eq!("test", table.table_name());
        assert_eq!(schema, InformationTable::schema(&table));

        let stream = table.to_stream().unwrap();

        let batches = RecordBatches::try_collect(stream).await.unwrap();

        assert_eq!(
            "\
+----+----+
| a  | b  |
+----+----+
| a1 | b1 |
| a2 | b2 |
+----+----+",
            batches.pretty_print().unwrap()
        );
    }

    #[tokio::test]
    async fn test_empty_memory_table() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]));

        let table = MemoryTable::new(42, "test", schema.clone(), vec![]);

        assert_eq!(42, table.table_id());
        assert_eq!("test", table.table_name());
        assert_eq!(schema, InformationTable::schema(&table));

        let stream = table.to_stream().unwrap();

        let batches = RecordBatches::try_collect(stream).await.unwrap();

        assert_eq!(
            "\
+---+---+
| a | b |
+---+---+
+---+---+",
            batches.pretty_print().unwrap()
        );
    }
}
