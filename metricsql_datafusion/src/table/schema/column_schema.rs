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
use std::fmt;

use arrow::datatypes::Field;
use arrow_schema::DataType;
use serde::{Deserialize, Serialize};

use crate::table::error::{Error, Result};
use crate::table::is_timestamp_field;

pub type Metadata = HashMap<String, String>;

/// Key used to store whether the column is time index in arrow field's metadata.
pub const TIME_INDEX_KEY: &str = "greptime:time_index";
pub const COMMENT_KEY: &str = "greptime:storage:comment";
/// Key used to store default constraint in arrow field's metadata.
const DEFAULT_CONSTRAINT_KEY: &str = "greptime:default_constraint";

/// Schema of a column, used as an immutable struct.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: DataType,
    is_nullable: bool,
    is_time_index: bool,
    metadata: Metadata,
}

impl fmt::Debug for ColumnSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.name,
            self.data_type,
            if self.is_nullable { "null" } else { "not null" },
        )?;

        // Add metadata if present
        if !self.metadata.is_empty() {
            write!(f, " metadata={:?}", self.metadata)?;
        }

        Ok(())
    }
}

impl ColumnSchema {
    pub fn new<T: Into<String>>(
        name: T,
        data_type: DataType,
        is_nullable: bool,
    ) -> ColumnSchema {
        ColumnSchema {
            name: name.into(),
            data_type,
            is_nullable,
            is_time_index: false,
            metadata: Metadata::new(),
        }
    }

    #[inline]
    pub fn is_time_index(&self) -> bool {
        self.is_time_index
    }

    #[inline]
    pub fn is_nullable(&self) -> bool {
        self.is_nullable
    }

    #[inline]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    #[inline]
    pub fn mut_metadata(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    pub fn with_time_index(mut self, is_time_index: bool) -> Self {
        self.is_time_index = is_time_index;
        if is_time_index {
            let _ = self
                .metadata
                .insert(TIME_INDEX_KEY.to_string(), "true".to_string());
        } else {
            let _ = self.metadata.remove(TIME_INDEX_KEY);
        }
        self
    }

    /// Set the nullablity to `true` of the column.
    pub fn with_nullable_set(mut self) -> Self {
        self.is_nullable = true;
        self
    }

    /// Creates a new [`ColumnSchema`] with given metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<&Field> for ColumnSchema {
    fn from(field: &Field) -> ColumnSchema {
        let data_type = field.data_type().clone();
        let mut metadata = field.metadata().clone();
        let is_time_index = is_timestamp_field(field);

        ColumnSchema {
            name: field.name().clone(),
            data_type,
            is_nullable: field.is_nullable(),
            is_time_index,
            metadata,
        }
    }
}

impl TryFrom<&ColumnSchema> for Field {
    type Error = Error;

    fn try_from(column_schema: &ColumnSchema) -> Result<Field> {
        let mut metadata = column_schema.metadata.clone();

        Ok(Field::new(
            &column_schema.name,
            column_schema.data_type.clone(),
            column_schema.is_nullable(),
        )
            .with_metadata(metadata))
    }
}

pub fn is_timestamp_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Timestamp(_, _) | DataType::Date32 | DataType::Date64
    )
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType as ArrowDataType;

    use super::*;

    #[test]
    fn test_column_schema() {
        let column_schema = ColumnSchema::new("test", DataType::Int32, true);
        let field = Field::try_from(&column_schema).unwrap();
        assert_eq!("test", field.name());
        assert_eq!(ArrowDataType::Int32, *field.data_type());
        assert!(field.is_nullable());

        let new_column_schema = ColumnSchema::try_from(&field).unwrap();
        assert_eq!(column_schema, new_column_schema);
    }

    #[test]
    fn test_column_schema_with_metadata() {
        let metadata = Metadata::from([("k1".to_string(), "v1".to_string())]);
        let column_schema = ColumnSchema::new("test", DataType::Int32, true)
            .with_metadata(metadata)
            .unwrap();
        assert_eq!("v1", column_schema.metadata().get("k1").unwrap());
        assert!(column_schema
            .metadata()
            .get(DEFAULT_CONSTRAINT_KEY)
            .is_none());

        let field = Field::try_from(&column_schema).unwrap();
        assert_eq!("v1", field.metadata().get("k1").unwrap());
        let _ = field.metadata().get(DEFAULT_CONSTRAINT_KEY).unwrap();

        let new_column_schema = ColumnSchema::try_from(&field).unwrap();
        assert_eq!(column_schema, new_column_schema);
    }

    #[test]
    fn test_column_schema_with_duplicate_metadata() {
        let metadata = Metadata::from([(DEFAULT_CONSTRAINT_KEY.to_string(), "v1".to_string())]);
        let column_schema = ColumnSchema::new("test", DataType::Int32, true)
            .with_metadata(metadata)
            .unwrap();
        assert!(Field::try_from(&column_schema).is_err());
    }

    #[test]
    fn test_column_schema_create_default_null() {
        // Implicit default null.
        let column_schema = ColumnSchema::new("test", DataType::Int32, true);
        let v = column_schema.create_default_vector(5).unwrap().unwrap();
        assert_eq!(5, v.len());
        assert!(v.only_null());

        // Explicit default null.
        let column_schema = ColumnSchema::new("test", DataType::Int32, true)
            .unwrap();
        let v = column_schema.create_default_vector(5).unwrap().unwrap();
        assert_eq!(5, v.len());
        assert!(v.only_null());
    }

    #[test]
    fn test_column_schema_no_default() {
        let column_schema = ColumnSchema::new("test", DataType::Int32, false);
        assert!(column_schema.create_default_vector(5).unwrap().is_none());
    }

    #[test]
    fn test_column_schema_single_no_default() {
        let column_schema = ColumnSchema::new("test", DataType::Int32, false);
        assert!(column_schema.create_default().unwrap().is_none());
    }

    #[test]
    fn test_debug_for_column_schema() {
        let column_schema_int8 =
            ColumnSchema::new("test_column_1", DataType::Int8, true);

        let column_schema_int32 =
            ColumnSchema::new("test_column_2", DataType::Int32, false);

        let formatted_int8 = format!("{:?}", column_schema_int8);
        let formatted_int32 = format!("{:?}", column_schema_int32);
        assert_eq!(formatted_int8, "test_column_1 Int8 null");
        assert_eq!(formatted_int32, "test_column_2 Int32 not null");
    }
}
