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
use std::slice;
use std::sync::Arc;

use arrow::array::{ArrayRef};
use arrow::compute;
use arrow_schema::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch as DfRecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_common::ScalarValue;
use itertools::Itertools;
use serde::ser::{Error, SerializeStruct};
use serde::{Serialize, Serializer};
use snafu::ResultExt;

use crate::common::recordbatch::error::{
    CastVectorSnafu, ColumnNotExistsSnafu, DataTypesSnafu, NewDfRecordBatchSnafu,
    ProjectArrowRecordBatchSnafu, Result,
};

/// A two-dimensional batch of column-oriented data with a defined schema.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordBatch {
    pub schema: SchemaRef,
    columns: Vec<ArrayRef>,
    df_record_batch: DfRecordBatch,
}

impl RecordBatch {
    /// Create a new [`RecordBatch`] from `schema` and `columns`.
    pub fn new<I: IntoIterator<Item = ArrayRef>>(
        schema: SchemaRef,
        columns: I,
    ) -> Result<RecordBatch> {
        let columns: Vec<_> = columns.into_iter().collect();

        // todo: how to avoid columns.clone() ?
        let df_record_batch = DfRecordBatch::try_new(schema.clone(), columns.clone())
            .context(NewDfRecordBatchSnafu)?;

        Ok(RecordBatch {
            schema,
            columns,
            df_record_batch,
        })
    }

    /// Create an empty [`RecordBatch`] from `schema`.
    pub fn new_empty(schema: SchemaRef) -> Result<RecordBatch> {
        let df_record_batch = DfRecordBatch::new_empty(schema.clone());
        Ok(RecordBatch {
            schema,
            columns: vec![],
            df_record_batch,
        })
    }

    pub fn try_project(&self, indices: &[usize]) -> Result<Self> {
        let schema = Arc::new(self.schema.project(indices).context(DataTypesSnafu)?);
        let mut columns = Vec::with_capacity(indices.len());
        for index in indices {
            columns.push(self.columns[*index].clone());
        }
        let df_record_batch = self.df_record_batch.project(indices).with_context(|_| {
            ProjectArrowRecordBatchSnafu {
                schema: self.schema.clone(),
                projection: indices.to_vec(),
            }
        })?;

        Ok(Self {
            schema,
            columns,
            df_record_batch,
        })
    }

    /// Create a new [`RecordBatch`] from `schema` and `df_record_batch`.
    ///
    /// This method doesn't check the schema.
    pub fn try_from_df_record_batch(
        schema: SchemaRef,
        df_record_batch: DfRecordBatch,
    ) -> Result<RecordBatch> {
        let columns = df_record_batch
            .columns()
            .iter()
            .cloned()
            .collect::<Vec<_>>();

        Ok(RecordBatch {
            schema,
            columns,
            df_record_batch,
        })
    }

    #[inline]
    pub fn df_record_batch(&self) -> &DfRecordBatch {
        &self.df_record_batch
    }

    #[inline]
    pub fn into_df_record_batch(self) -> DfRecordBatch {
        self.df_record_batch
    }

    #[inline]
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    #[inline]
    pub fn column(&self, idx: usize) -> &ArrayRef {
        &self.columns[idx]
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        if let Some((idx, _)) = self.schema.fields.iter().find_position(|f| f.name() == name) {
            Some(&self.columns[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.df_record_batch.num_rows()
    }

    /// Create an iterator to traverse the data by row
    pub fn rows(&self) -> RecordBatchRowIterator<'_> {
        RecordBatchRowIterator::new(self)
    }

    pub fn column_vectors(
        &self,
        table_name: &str,
        table_schema: SchemaRef,
    ) -> Result<HashMap<String, ArrayRef>> {
        let mut vectors = HashMap::with_capacity(self.num_columns());

        // column schemas in recordbatch must match its vectors, otherwise it's corrupted
        for (vector_schema, vector) in self.schema.fields().iter().zip(self.columns.iter())
        {
            let column_name = vector_schema.name();
            let column_schema =
                table_schema
                    .field_with_name(column_name)
                    .context(ColumnNotExistsSnafu {
                        table_name: table_name.to_string(),
                        column_name: column_name.to_string(),
                    })?;
            let vector = if vector_schema.data_type() != column_schema.data_type() {
                compute::cast(vector, &column_schema.data_type())
                    .with_context(|_| CastVectorSnafu {
                        from_type: vector_schema.data_type(),
                        to_type: column_schema.data_type(),
                    })?
            } else {
                vector.clone()
            };

            let _ = vectors.insert(column_name.clone(), vector);
        }

        Ok(vectors)
    }

    /// Pretty display this record batch like a table
    pub fn pretty_print(&self) -> String {
        pretty_format_batches(slice::from_ref(&self.df_record_batch))
            .map(|t| t.to_string())
            .unwrap_or("failed to pretty display a record batch".to_string())
    }
}

impl Serialize for RecordBatch {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // TODO(yingwen): arrow and arrow2's schemas have different fields, so
        // it might be better to use our `RawSchema` as serialized field.
        let mut s = serializer.serialize_struct("record", 2)?;
        s.serialize_field("schema", &self.schema)?;

        let vec = self
            .columns
            .iter()
            .map(|c| c.serialize_to_json())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(S::Error::custom)?;

        s.serialize_field("columns", &vec)?;
        s.end()
    }
}

pub struct RecordBatchRowIterator<'a> {
    record_batch: &'a RecordBatch,
    rows: usize,
    columns: usize,
    row_cursor: usize,
}

impl<'a> RecordBatchRowIterator<'a> {
    fn new(record_batch: &'a RecordBatch) -> RecordBatchRowIterator {
        RecordBatchRowIterator {
            record_batch,
            rows: record_batch.df_record_batch.num_rows(),
            columns: record_batch.df_record_batch.num_columns(),
            row_cursor: 0,
        }
    }
}

impl<'a> Iterator for RecordBatchRowIterator<'a> {
    type Item = Vec<ScalarValue>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_cursor == self.rows {
            None
        } else {
            let mut row = Vec::with_capacity(self.columns);

            for col in 0..self.columns {
                let column = self.record_batch.column(col);
                let value = column.as_ref()
                    .value(self.row_cursor);

                row.push(value);
            }

            self.row_cursor += 1;
            Some(row.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::ScalarValue;

    use crate::table::schema::column_schema::Field;

    use super::*;

    #[test]
    fn test_record_batch() {
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::UInt32, false),
            Field::new("c2", DataType::UInt32, false),
        ]));
        let schema = Arc::new(Schema::try_from(arrow_schema).unwrap());

        let c1 = Arc::new(UInt32Array::from_slice([1, 2, 3]));
        let c2 = Arc::new(UInt32Array::from_slice([4, 5, 6]));
        let columns: Vec<ArrayRef> = vec![c1, c2];

        let batch = RecordBatch::new(schema.clone(), columns.clone()).unwrap();
        assert_eq!(3, batch.num_rows());
        assert_eq!(&columns, batch.columns());
        for (i, expect) in columns.iter().enumerate().take(batch.num_columns()) {
            let column = batch.column(i);
            assert_eq!(expect, column);
        }
        assert_eq!(schema, batch.schema);

        assert_eq!(columns[0], *batch.column_by_name("c1").unwrap());
        assert_eq!(columns[1], *batch.column_by_name("c2").unwrap());
        assert!(batch.column_by_name("c3").is_none());

        let converted =
            RecordBatch::try_from_df_record_batch(schema, batch.df_record_batch().clone()).unwrap();
        assert_eq!(batch, converted);
        assert_eq!(*batch.df_record_batch(), converted.into_df_record_batch());
    }

    #[test]
    pub fn test_serialize_recordbatch() {
        let column_schemas = vec![Field::new("number", DataType::UInt32, false)];
        let schema = Arc::new(Schema::try_new(column_schemas).unwrap());

        let numbers: Vec<u32> = (0..10).collect();
        let columns = vec![Arc::new(UInt32Array::from_slice(numbers)) as ArrayRef];
        let batch = RecordBatch::new(schema, columns).unwrap();

        let output = serde_json::to_string(&batch).unwrap();
        assert_eq!(
            r#"{"schema":{"fields":[{"name":"number","data_type":"UInt32","nullable":false,"dict_id":0,"dict_is_ordered":false,"metadata":{}}],"metadata":{"greptime:version":"0"}},"columns":[[0,1,2,3,4,5,6,7,8,9]]}"#,
            output
        );
    }

    #[test]
    fn test_record_batch_visitor() {
        let column_schemas = vec![
            Field::new("numbers", DataType::UInt32, false),
            Field::new("strings", DataType::Utf8, true),
        ];
        let schema = Arc::new(Schema::new(column_schemas));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from_slice(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                None,
                Some("hello"),
                Some("greptime"),
                None,
            ])),
        ];
        let record_batch = RecordBatch::new(schema, columns).unwrap();

        let mut record_batch_iter = record_batch.rows();
        assert_eq!(
            vec![ScalarValue::UInt32(Some(1)), ScalarValue::Null],
            record_batch_iter
                .next()
                .unwrap()
                .into_iter()
                .collect::<Vec<ScalarValue>>()
        );

        assert_eq!(
            vec![
                ScalarValue::UInt32(Some(2)),
                ScalarValue::Utf8("hello".into())
            ],
            record_batch_iter
                .next()
                .unwrap()
                .into_iter()
                .collect::<Vec<ScalarValue>>()
        );

        assert_eq!(
            vec![
                ScalarValue::UInt32(Some(3)),
                ScalarValue::Utf8("greptime".into())
            ],
            record_batch_iter
                .next()
                .unwrap()
                .into_iter()
                .collect::<Vec<ScalarValue>>()
        );

        assert_eq!(
            vec![ScalarValue::UInt32(Some(4)), ScalarValue::Null],
            record_batch_iter
                .next()
                .unwrap()
                .into_iter()
                .collect::<Vec<ScalarValue>>()
        );

        assert!(record_batch_iter.next().is_none());
    }
}
