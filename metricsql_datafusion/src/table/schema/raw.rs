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

use arrow_schema::{Schema, SchemaBuilder};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::table::error::{Error, Result};
use crate::table::{is_timestamp_field, TIMESTAMP_COLUMN_KEY};
use crate::table::schema::column_schema::ColumnSchema;

/// Struct used to serialize and deserialize [`Schema`](crate::schema::Schema).
///
/// This struct only contains necessary data to recover the Schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawSchema {
    /// Schema of columns.
    pub column_schemas: Vec<ColumnSchema>,
    /// Index of the timestamp column.
    pub timestamp_index: Option<usize>,
    /// Schema version.
    pub version: u32,
}

impl RawSchema {
    /// Creates a new [RawSchema] from specific `column_schemas`.
    ///
    /// Sets [RawSchema::timestamp_index] to the first index of the timestamp
    /// column. It doesn't check whether time index column is duplicate.
    pub fn new(column_schemas: Vec<ColumnSchema>) -> RawSchema {
        let timestamp_index = column_schemas
            .iter()
            .position(|column_schema| column_schema.is_time_index());

        RawSchema {
            column_schemas,
            timestamp_index,
            version: 0,
        }
    }
}

impl TryFrom<RawSchema> for Schema {
    type Error = Error;

    fn try_from(raw: RawSchema) -> Result<Schema> {
        // While building Schema, we don't trust the fields, such as timestamp_index,
        // in RawSchema. We use SchemaBuilder to perform the validation.
        let mut fields = raw.column_schemas
            .into_iter()
            .map(|column_schema| column_schema.into())
            .collect::<Vec<_>>();

        if let Some(timestamp_index) = raw.timestamp_index {
            if !is_timestamp_field(&fields[timestamp_index]) {
                return Err(Error::MissingTimeIndexColumn(format!(
                    "timestamp_index {} is not a timestamp field",
                    timestamp_index
                )));
            }
            fields[timestamp_index].metadata().insert(
                TIMESTAMP_COLUMN_KEY.to_string(),
                "true".to_string(),
            );
        }
        //Ok(Schema::new(fields))
        SchemaBuilder::try_from(fields)?
            .build()
    }
}

impl From<&Schema> for RawSchema {
    fn from(schema: &Schema) -> RawSchema {
        let timestamp_index = schema.fields.iter()
            .find_position(|field| is_timestamp_field(field))
            .map(|(i, _)| i);

        let column_schemas = schema.fields.iter().map(|field| {
            ColumnSchema::from(field.as_ref())
        }).collect::<Vec<_>>();

        RawSchema {
            column_schemas,
            timestamp_index,
            version: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, TimeUnit};

    use super::*;

    #[test]
    fn test_raw_convert() {
        let column_schemas = vec![
            ColumnSchema::new("col1", DataType::Int32, true),
            ColumnSchema::new(
                "ts",
                DataType::Time64(TimeUnit::Millisecond),
                false,
            )
                .with_time_index(true),
        ];
        let schema = SchemaBuilder::try_from(column_schemas)
            .unwrap()
            .version(123)
            .build()
            .unwrap();

        let raw = RawSchema::from(&schema);
        let schema_new = Schema::try_from(raw).unwrap();

        assert_eq!(schema, schema_new);
    }

    #[test]
    fn test_new_raw_schema_with_time_index() {
        let column_schemas = vec![
            ColumnSchema::new("col1", DataType::Int32, true),
            ColumnSchema::new(
                "ts",
                DataType::Time64(TimeUnit::Millisecond),
                false,
            )
                .with_time_index(true),
        ];
        let schema = RawSchema::new(column_schemas);
        assert_eq!(1, schema.timestamp_index.unwrap());
    }

    #[test]
    fn test_new_raw_schema_without_time_index() {
        let column_schemas = vec![
            ColumnSchema::new("col1", DataType::Int32, true),
            ColumnSchema::new(
                "ts",
                DataType::Time64(TimeUnit::Millisecond),
                false,
            ),
        ];
        let schema = RawSchema::new(column_schemas);
        assert!(schema.timestamp_index.is_none());
    }
}
