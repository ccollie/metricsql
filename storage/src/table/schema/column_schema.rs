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

use arrow::datatypes::Field;
use serde::{Deserialize, Serialize};
use snafu::ResultExt;

use crate::datatypes::data_type::ConcreteDataType;
use crate::error::{Error, Result};

pub type Metadata = HashMap<String, String>;

/// Key used to store whether the column is time index in arrow field's metadata.
pub const TIME_INDEX_KEY: &str = "greptime:time_index";
pub const COMMENT_KEY: &str = "greptime:storage:comment";
/// Key used to store default constraint in arrow field's metadata.
const DEFAULT_CONSTRAINT_KEY: &str = "greptime:default_constraint";

/// Schema of a column, used as an immutable struct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: ConcreteDataType,
    is_nullable: bool,
    is_time_index: bool,
    metadata: Metadata,
}

impl ColumnSchema {
    pub fn new<T: Into<String>>(
        name: T,
        data_type: ConcreteDataType,
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

    /// Creates a new [`ColumnSchema`] with given metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl TryFrom<&Field> for ColumnSchema {
    type Error = Error;

    fn try_from(field: &Field) -> Result<ColumnSchema> {
        let data_type = ConcreteDataType::try_from(field.data_type())?;
        let mut metadata = field.metadata().clone();
        let is_time_index = metadata.contains_key(TIME_INDEX_KEY);

        Ok(ColumnSchema {
            name: field.name().clone(),
            data_type,
            is_nullable: field.is_nullable(),
            is_time_index,
            metadata,
        })
    }
}

impl TryFrom<&ColumnSchema> for Field {
    type Error = Error;

    fn try_from(column_schema: &ColumnSchema) -> Result<Field> {
        let mut metadata = column_schema.metadata.clone();
        Ok(Field::new(
            &column_schema.name,
            column_schema.data_type.as_arrow_type(),
            column_schema.is_nullable(),
        )
        .with_metadata(metadata))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::DataType as ArrowDataType;

    use crate::value::Value;
    use crate::vectors::Int32Vector;

    use super::*;

    #[test]
    fn test_column_schema() {
        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), true);
        let field = Field::try_from(&column_schema).unwrap();
        assert_eq!("test", field.name());
        assert_eq!(ArrowDataType::Int32, *field.data_type());
        assert!(field.is_nullable());

        let new_column_schema = ColumnSchema::try_from(&field).unwrap();
        assert_eq!(column_schema, new_column_schema);
    }

    #[test]
    fn test_column_schema_with_duplicate_metadata() {
        let metadata = Metadata::from([(DEFAULT_CONSTRAINT_KEY.to_string(), "v1".to_string())]);
        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), true)
            .with_metadata(metadata)
            .unwrap();
        assert!(Field::try_from(&column_schema).is_err());
    }

    #[test]
    fn test_column_schema_no_default() {
        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), false);
        assert!(column_schema.create_default_vector(5).unwrap().is_none());
    }

    #[test]
    fn test_create_default_vector_for_padding() {
        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), true);
        let vector = column_schema.create_default_vector_for_padding(4);
        assert!(vector.only_null());
        assert_eq!(4, vector.len());

        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), false);
        let vector = column_schema.create_default_vector_for_padding(4);
        assert_eq!(4, vector.len());
        let expect: VectorRef = Arc::new(Int32Vector::from_slice([0, 0, 0, 0]));
        assert_eq!(expect, vector);
    }

    #[test]
    fn test_column_schema_single_no_default() {
        let column_schema = ColumnSchema::new("test", ConcreteDataType::int32_datatype(), false);
        assert!(column_schema.create_default().unwrap().is_none());
    }
}
