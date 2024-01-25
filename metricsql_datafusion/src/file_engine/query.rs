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

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::ArrayRef;
use arrow::compute;
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use datafusion::logical_expr::utils as df_logical_expr_utils;
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use futures::Stream;
use snafu::{ensure, ResultExt};

use metricsql_common::prelude::BoxedError;

use crate::common::recordbatch::{RecordBatch, RecordBatchStream, SendableRecordBatchStream};
use crate::common::recordbatch::error::{CastVectorSnafu, ExternalSnafu, Result as RecordBatchResult};
use crate::datasource::object_store::build_backend;
use crate::datatypes::try_array_from_scalar_value;
use crate::file_engine::error::{
    BuildBackendSnafu, CreateDefaultSnafu, ExtractColumnFromFilterSnafu,
    ProjectionOutOfBoundsSnafu, ProjectSchemaSnafu, Result
};
use crate::table::storage::ScanRequest;

use super::file_table::FileTable;

use self::file_stream::{CreateScanPlanContext, ScanPlanConfig};

pub(crate) mod file_stream;

impl FileTable {
    pub fn query(&self, request: ScanRequest) -> Result<SendableRecordBatchStream> {
        let store = build_backend(&self.url, &self.options).context(BuildBackendSnafu)?;

        let file_projection = self.projection_pushdown_to_file(&request.projection)?;
        let file_filters = self.filters_pushdown_to_file(&request.filters)?;
        let file_schema = Arc::new(Schema::new(self.file_options.file_column_schemas.clone()));

        let file_stream = file_stream::create_stream(
            &self.format,
            &CreateScanPlanContext::default(),
            &ScanPlanConfig {
                file_schema,
                files: &self.file_options.files,
                projection: file_projection.as_ref(),
                filters: &file_filters,
                limit: request.limit,
                store,
            },
        )?;

        let scan_schema = self.scan_schema(&request.projection)?;

        Ok(Box::pin(FileToScanRegionStream::new(
            scan_schema,
            file_stream,
        )))
    }

    fn projection_pushdown_to_file(
        &self,
        req_projection: &Option<Vec<usize>>,
    ) -> Result<Option<Vec<usize>>> {
        let Some(scan_projection) = req_projection.as_ref() else {
            return Ok(None);
        };

        let file_column_schemas = &self.file_options.file_column_schemas;
        let mut file_projection = Vec::with_capacity(scan_projection.len());
        let schema = self.schema();

        for column_index in scan_projection {
            let field = schema.fields.get(*column_index);
            ensure!(
                field.is_some(),
                ProjectionOutOfBoundsSnafu {
                    column_index: *column_index,
                    bounds: schema.fields.len()
                }
            );

            let column_name = field.unwrap().name();
            let file_column_index = file_column_schemas
                .iter()
                .position(|c| c.name() == column_name);
            if let Some(file_column_index) = file_column_index {
                file_projection.push(file_column_index);
            }
        }
        Ok(Some(file_projection))
    }

    // Collects filters that can be pushed down to the file, specifically filters where Expr
    // only contains columns from the file.
    fn filters_pushdown_to_file(&self, scan_filters: &[Expr]) -> Result<Vec<Expr>> {
        let mut file_filters = Vec::with_capacity(scan_filters.len());

        let file_column_names = self
            .file_options
            .file_column_schemas
            .iter()
            .map(|c| c.name())
            .collect::<HashSet<_>>();

        let mut aux_column_set = HashSet::new();
        for scan_filter in scan_filters {
            df_logical_expr_utils::expr_to_columns(scan_filter, &mut aux_column_set)
                .context(ExtractColumnFromFilterSnafu)?;

            let all_file_columns = aux_column_set
                .iter()
                .all(|column_in_expr| file_column_names.contains(&column_in_expr.name));

            if all_file_columns {
                file_filters.push(scan_filter.clone());
            }
            aux_column_set.clear();
        }
        Ok(file_filters)
    }

    fn scan_schema(&self, req_projection: &Option<Vec<usize>>) -> Result<SchemaRef> {
        let schema = if let Some(indices) = req_projection {
            Arc::new(
                self
                    .schema()
                    .project(indices)
                    .context(ProjectSchemaSnafu)?,
            )
        } else {
            self.schema().clone()
        };

        Ok(schema)
    }
}

struct FileToScanRegionStream {
    scan_schema: SchemaRef,
    file_stream: SendableRecordBatchStream,
}

impl RecordBatchStream for FileToScanRegionStream {
    fn schema(&self) -> SchemaRef {
        self.scan_schema.clone()
    }
}

impl Stream for FileToScanRegionStream {
    type Item = RecordBatchResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.file_stream).poll_next(ctx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(file_record_batch)) => {
                let file_record_batch = file_record_batch?;
                let scan_record_batch = if self.schema_eq(&file_record_batch) {
                    Ok(file_record_batch)
                } else {
                    self.convert_record_batch(&file_record_batch)
                };

                Poll::Ready(Some(scan_record_batch))
            }
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

impl FileToScanRegionStream {
    fn new(scan_schema: SchemaRef, file_stream: SendableRecordBatchStream) -> Self {
        Self {
            scan_schema,
            file_stream,
        }
    }

    fn schema_eq(&self, file_record_batch: &RecordBatch) -> bool {
        self.scan_schema
            .fields
            .iter()
            .all(|scan_column_schema| {
                file_record_batch
                    .column_by_name(&scan_column_schema.name())
                    .map(|rb| rb.data_type() == scan_column_schema.data_type())
                    .unwrap_or_default()
            })
    }

    /// Converts a RecordBatch from file schema to scan schema.
    ///
    /// This function performs the following operations:
    /// - Projection: Only columns present in scan schema are retained.
    /// - Cast Type: Columns present in both file schema and scan schema but with different types are cast to the type in scan schema.
    /// - Backfill: Columns present in scan schema but not in file schema are backfilled with default values.
    fn convert_record_batch(
        &self,
        file_record_batch: &RecordBatch,
    ) -> RecordBatchResult<RecordBatch> {
        let file_row_count = file_record_batch.num_rows();
        let columns = self
            .scan_schema
            .fields
            .iter()
            .map(|scan_column_schema| {
                let file_column = file_record_batch.column_by_name(&scan_column_schema.name());
                if let Some(file_column) = file_column {
                    Self::cast_column_type(file_column, &scan_column_schema.data_type())
                } else {
                    Self::backfill_column(scan_column_schema, file_row_count)
                }
            })
            .collect::<RecordBatchResult<Vec<_>>>()?;

        RecordBatch::new(self.scan_schema.clone(), columns)
    }

    fn cast_column_type(
        source_column: &ArrayRef,
        target_data_type: &DataType,
    ) -> RecordBatchResult<ArrayRef> {
        let source_type = source_column.data_type();
        if source_type == target_data_type {
            Ok(source_column.clone())
        } else {
            compute::cast(source_column, target_data_type)
                .context(CastVectorSnafu {
                    from_type: source_type.clone(),
                    to_type: target_data_type.clone(),
                })
        }
    }

    fn backfill_column(
        column_schema: &FieldRef,
        num_rows: usize,
    ) -> RecordBatchResult<ArrayRef> {
        Self::create_default_vector(column_schema, num_rows)
            .map_err(BoxedError::new)
            .context(ExternalSnafu)
    }

    fn create_default_vector(column_schema: &FieldRef, num_rows: usize) -> Result<ArrayRef> {
        let data_type = column_schema.data_type();
        let value = ScalarValue::new_zero(data_type)
            .with_context(|_| CreateDefaultSnafu {
                column: column_schema.name().clone(),
            })?;

        Ok(try_array_from_scalar_value(value, num_rows)
            .with_context(|_| CreateDefaultSnafu {
                column: column_schema.name().clone(),
            })?)
    }
}
