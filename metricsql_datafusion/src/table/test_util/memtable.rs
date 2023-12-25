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

use std::pin::Pin;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_array::{ArrayRef, UInt32Array};
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_expr::TableType;
use futures::Stream;
use futures::task::{Context, Poll};
use snafu::prelude::*;

use crate::catalog::consts::{DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME};
use crate::common::error::ext::BoxedError;
use crate::common::recordbatch::RecordBatchStream;
use crate::data_source::DataSource;
use crate::table::{
    FilterPushDownType, TableId, TableInfoBuilder, TableMetaBuilder, TableRef, TableVersion, ThinTable,
    ThinTableAdapter,
};
use crate::table::error::{SchemaConversionSnafu, TableProjectionSnafu, TablesRecordBatchSnafu};
use crate::table::numbers::RecordBatchResult;
use crate::table::storage::ScanRequest;

pub struct MemTable;

impl MemTable {
    pub fn table(table_name: impl Into<String>, record_batch: RecordBatch) -> TableRef {
        Self::new_with_region(table_name, record_batch)
    }

    pub fn new_with_region(table_name: impl Into<String>, record_batch: RecordBatch) -> TableRef {
        Self::new_with_catalog(
            table_name,
            record_batch,
            1,
            DEFAULT_CATALOG_NAME.to_string(),
            DEFAULT_SCHEMA_NAME.to_string(),
        )
    }

    pub fn new_with_catalog(
        table_name: impl Into<String>,
        record_batch: RecordBatch,
        table_id: TableId,
        catalog_name: String,
        schema_name: String,
    ) -> TableRef {
        let schema = record_batch.schema().clone();

        let meta = TableMetaBuilder::default()
            .schema(schema)
            .primary_key_indices(vec![])
            .value_indices(vec![])
            .engine("mito".to_string())
            .next_column_id(0)
            .options(Default::default())
            .created_on(Default::default())
            .build()
            .unwrap();

        let info = Arc::new(
            TableInfoBuilder::default()
                .table_id(table_id)
                .table_version(0 as TableVersion)
                .name(table_name.into())
                .schema_name(schema_name)
                .catalog_name(catalog_name)
                .desc(None)
                .table_type(TableType::Base)
                .meta(meta)
                .build()
                .unwrap(),
        );

        let thin_table = ThinTable::new(info, FilterPushDownType::Unsupported);
        let data_source = Arc::new(MemtableDataSource { record_batch });
        Arc::new(ThinTableAdapter::new(thin_table, data_source))
    }

    /// Creates a 1 column 100 rows table, with table name "numbers", column name "uint32s" and
    /// column type "uint32". Column data increased from 0 to 100.
    pub fn default_numbers_table() -> TableRef {
        let column_schemas = vec![Field::new("uint32s", Field::UInt32, true)];
        let schema = Arc::new(Schema::new(column_schemas));
        let columns: Vec<ArrayRef> = vec![Arc::new(UInt32Array::from_slice(
            (0..100).collect::<Vec<_>>(),
        ))];
        let record_batch = RecordBatch::new(schema, columns).unwrap();
        MemTable::table("numbers", record_batch)
    }
}

struct MemtableDataSource {
    record_batch: RecordBatch,
}

impl DataSource for MemtableDataSource {
    fn get_stream(&self, request: ScanRequest) -> Result<SendableRecordBatchStream, BoxedError> {
        let record_batch = if let Some(indices) = request.projection {
            self.record_batch
                .df_record_batch()
                .project(&indices)
                .context(TableProjectionSnafu)
                .map_err(BoxedError::new)?
        } else {
            self.record_batch.df_record_batch().clone()
        };

        let rows = record_batch.num_rows();
        let limit = if let Some(limit) = request.limit {
            limit.min(rows)
        } else {
            rows
        };
        let df_recordbatch = record_batch.slice(0, limit);

        let record_batch = RecordBatch::try_from_df_record_batch(
            Arc::new(
                Schema::try_from(df_recordbatch.schema())
                    .context(SchemaConversionSnafu)
                    .map_err(BoxedError::new)?,
            ),
            df_recordbatch,
        )
        .map_err(BoxedError::new)
        .context(TablesRecordBatchSnafu)
        .map_err(BoxedError::new)?;

        Ok(Box::pin(MemtableStream {
            schema: record_batch.schema.clone(),
            record_batch: Some(record_batch),
        }))
    }
}

impl RecordBatchStream for MemtableStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

struct MemtableStream {
    schema: SchemaRef,
    record_batch: Option<RecordBatch>,
}

impl Stream for MemtableStream {
    type Item = RecordBatchResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.record_batch.take() {
            Some(records) => Poll::Ready(Some(Ok(records))),
            None => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::DataType;

    use crate::common::recordbatch::util;

    use super::*;

    #[tokio::test]
    async fn test_scan_with_projection() {
        let table = build_testing_table();

        let scan_req = ScanRequest {
            projection: Some(vec![1]),
            ..Default::default()
        };
        let stream = table.scan_to_stream(scan_req).await.unwrap();
        let record_batch = util::collect(stream).await.unwrap();
        assert_eq!(1, record_batch.len());
        let columns = record_batch[0].df_record_batch().columns();
        assert_eq!(1, columns.len());

        let string_column = &columns[0].as_any().downcast_ref::<StringArray>().unwrap();
        let string_column = string_column.iter_data().flatten().collect::<Vec<&str>>();
        assert_eq!(vec!["hello", "greptime"], string_column);
    }

    #[tokio::test]
    async fn test_scan_with_limit() {
        let table = build_testing_table();

        let scan_req = ScanRequest {
            limit: Some(2),
            ..Default::default()
        };
        let stream = table.scan_to_stream(scan_req).await.unwrap();
        let recordbatch = util::collect(stream).await.unwrap();
        assert_eq!(1, recordbatch.len());
        let columns = recordbatch[0].df_record_batch().columns();
        assert_eq!(2, columns.len());

        let i32_column = &columns[0].as_any().downcast_ref::<Int32Array>().unwrap();
        let i32_column = i32_column.iter_data().flatten().collect::<Vec<i32>>();
        assert_eq!(vec![-100], i32_column);

        let string_column = &columns[1].as_any().downcast_ref::<StringArray>().unwrap();
        let string_column = string_column.iter_data().flatten().collect::<Vec<&str>>();
        assert_eq!(vec!["hello"], string_column);
    }

    fn build_testing_table() -> TableRef {
        let i32_column_schema = Field::new("i32_numbers", DataType::Int32, true);
        let string_column_schema = Field::new("strings", DataType::Utf8, true);
        let column_schemas = vec![i32_column_schema, string_column_schema];

        let schema = Arc::new(Schema::new(column_schemas));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![Some(-100), None, Some(1), Some(100)])),
            Arc::new(StringArray::from(vec![
                Some("hello"),
                None,
                Some("greptime"),
                None,
            ])),
        ];
        let record_batch = RecordBatch::new(schema, columns).unwrap();
        MemTable::table("", record_batch)
    }
}
