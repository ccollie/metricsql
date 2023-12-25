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
use std::sync::Arc;
use arrow_schema::DataType;

use chrono::{DateTime, Utc};
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion_expr::TableType;
use derive_builder::Builder;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::catalog::consts::{DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME};
pub use crate::datatypes::error::{Error as ConvertError, Result as ConvertResult};
use crate::table::requests::TableOptions;
use crate::table::schema::raw::RawSchema;

pub type TableId = u32;
pub type TableVersion = u64;

/// Indicates whether and how a filter expression can be handled by a
/// Table for table scans.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum FilterPushDownType {
    /// The expression cannot be used by the provider.
    Unsupported,
    /// The expression can be used to help minimise the data retrieved,
    /// but the provider cannot guarantee that all returned tuples
    /// satisfy the filter. The Filter plan node containing this expression
    /// will be preserved.
    Inexact,
    /// The provider guarantees that all returned data satisfies this
    /// filter expression. The Filter plan node containing this expression
    /// will be removed.
    Exact,
}

impl From<TableProviderFilterPushDown> for FilterPushDownType {
    fn from(value: TableProviderFilterPushDown) -> Self {
        match value {
            TableProviderFilterPushDown::Unsupported => FilterPushDownType::Unsupported,
            TableProviderFilterPushDown::Inexact => FilterPushDownType::Inexact,
            TableProviderFilterPushDown::Exact => FilterPushDownType::Exact,
        }
    }
}

impl From<FilterPushDownType> for TableProviderFilterPushDown {
    fn from(value: FilterPushDownType) -> Self {
        match value {
            FilterPushDownType::Unsupported => TableProviderFilterPushDown::Unsupported,
            FilterPushDownType::Inexact => TableProviderFilterPushDown::Inexact,
            FilterPushDownType::Exact => TableProviderFilterPushDown::Exact,
        }
    }
}

/// Identifier of the table.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
pub struct TableIdent {
    /// Unique id of this table.
    pub table_id: TableId,
    /// Version of the table, bumped when metadata (such as schema) of the table
    /// being changed.
    pub version: TableVersion,
}

/// The table metadata
/// Note: if you add new fields to this struct, please ensure 'new_meta_builder' function works.
/// TODO(dennis): find a better way to ensure 'new_meta_builder' works when adding new fields.
#[derive(Clone, Debug, Hash, Builder, PartialEq, Eq)]
pub struct TableMeta {
    pub schema: SchemaRef,
    /// The indices of columns in primary key. Note that the index of timestamp column
    /// is not included in these indices.
    pub primary_key_indices: Vec<usize>,
    /// The indices of columns to return as tags in query responses
    pub tag_column_indices: Option<Vec<usize>>,
    #[builder(default, setter(into))]
    pub engine: String,
    /// Options for table query_engine.
    #[builder(default)]
    pub engine_options: HashMap<String, String>,
    /// Table options.
    #[builder(default)]
    pub options: TableOptions,
    #[builder(default = "Utc::now()")]
    pub created_on: DateTime<Utc>,
    #[builder(default, setter(into))]
    pub timestamp_column: String,
    #[builder(default, setter(into))]
    pub value_column: String,
}

impl TableMetaBuilder {
    pub fn new_external_table() -> Self {
        Self {
            primary_key_indices: Some(Vec::new()),
            ..Default::default()
        }
    }
}

impl TableMeta {
    pub fn row_key_column_names(&self) -> impl Iterator<Item = &String> {
        let columns_schemas = &self.schema.fields;
        self.primary_key_indices
            .iter()
            .map(|idx| columns_schemas[*idx].name())
    }

    pub fn field_column_names(&self) -> impl Iterator<Item = &String> {
        // `value_indices` is wrong under distributed mode. Use the logic copied from DESC TABLE
        let primary_key_indices = &self.primary_key_indices;
        self.schema
            .fields
            .iter()
            .enumerate()
            .filter(|(i, cs)| !primary_key_indices.contains(i) && !cs.is_time_index())
            .map(|(_, cs)| cs.name())
    }

    pub fn tag_column_names(&self) -> impl Iterator<Item = &String> {
        let columns_schemas = &self.schema.fields;
        if let Some(tag_column_indices) = &self.tag_column_indices {
            tag_column_indices
                .iter()
                .map(|idx| columns_schemas[*idx].name())
        } else {
            let columns_schemas = &self.schema.fields;
            self.primary_key_indices
                .iter()
                .map(|idx| columns_schemas[*idx].name())
        }
    }

    fn new_meta_builder(&self) -> TableMetaBuilder {
        let mut builder = TableMetaBuilder::default();
        let _ = builder
            .engine(&self.engine)
            .engine_options(self.engine_options.clone())
            .options(self.options.clone())
            .created_on(self.created_on);

        builder
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Builder)]
#[builder(pattern = "owned")]
pub struct TableInfo {
    /// Id and version of the table.
    #[builder(default, setter(into))]
    pub ident: TableIdent,
    /// Name of the table.
    #[builder(setter(into))]
    pub name: String,
    /// Comment of the table.
    #[builder(default, setter(into))]
    pub desc: Option<String>,
    #[builder(default = "DEFAULT_CATALOG_NAME.to_string()", setter(into))]
    pub catalog_name: String,
    #[builder(default = "DEFAULT_SCHEMA_NAME.to_string()", setter(into))]
    pub schema_name: String,
    pub meta: TableMeta,
    #[builder(default = "TableType::Base")]
    pub table_type: TableType,
    #[builder(default, setter(into))]
    pub timestamp_column: String,
    #[builder(default, setter(into))]
    pub value_column: String,
}

impl TableInfo {
    pub fn ts_column_name(&self) -> &str {
        &self.meta.timestamp_column
    }
    pub fn value_column_name(&self) -> &str {
        &self.meta.value_column
    }
    pub fn schema(&self) -> SchemaRef {
        self.meta.schema.clone()
    }
}

pub type TableInfoRef = Arc<TableInfo>;

impl TableInfoBuilder {
    pub fn new<S: Into<String>>(name: S, meta: TableMeta) -> Self {
        Self {
            name: Some(name.into()),
            meta: Some(meta),
            ..Default::default()
        }
    }

    pub fn table_id(mut self, id: TableId) -> Self {
        let ident = self.ident.get_or_insert_with(TableIdent::default);
        ident.table_id = id;
        self
    }

    pub fn table_version(mut self, version: TableVersion) -> Self {
        let ident = self.ident.get_or_insert_with(TableIdent::default);
        ident.version = version;
        self
    }
}

impl TableIdent {
    pub fn new(table_id: TableId) -> Self {
        Self {
            table_id,
            version: 0,
        }
    }
}

impl From<TableId> for TableIdent {
    fn from(table_id: TableId) -> Self {
        Self::new(table_id)
    }
}

/// Struct used to serialize and deserialize [`TableMeta`].
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct RawTableMeta {
    pub schema: RawSchema,
    pub primary_key_indices: Vec<usize>,
    pub tag_column_indices: Option<Vec<usize>>,
    pub engine: String,
    pub engine_options: HashMap<String, String>,
    pub options: TableOptions,
    pub created_on: DateTime<Utc>,
    pub timestamp_column: String,
    pub value_column: String,
}

impl From<TableMeta> for RawTableMeta {
    fn from(meta: TableMeta) -> RawTableMeta {
        RawTableMeta {
            schema: RawSchema::from(&*meta.schema),
            primary_key_indices: meta.primary_key_indices,
            tag_column_indices: meta.tag_column_indices,
            engine: meta.engine,
            engine_options: meta.engine_options,
            options: meta.options,
            created_on: meta.created_on,
            timestamp_column: meta.timestamp_column,
            value_column: meta.value_column,
        }
    }
}

impl TryFrom<RawTableMeta> for TableMeta {
    type Error = ConvertError;

    fn try_from(raw: RawTableMeta) -> ConvertResult<TableMeta> {
        Ok(TableMeta {
            schema: Arc::new(Schema::try_from(raw.schema)?),
            primary_key_indices: raw.primary_key_indices,
            tag_column_indices: raw.tag_column_indices,
            engine: raw.engine,
            engine_options: raw.engine_options,
            options: raw.options,
            created_on: raw.created_on,
            timestamp_column: raw.timestamp_column,
            value_column: raw.value_column,
        })
    }
}

/// Struct used to serialize and deserialize [`TableInfo`].
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct RawTableInfo {
    pub ident: TableIdent,
    pub name: String,
    pub desc: Option<String>,
    pub catalog_name: String,
    pub schema_name: String,
    pub meta: RawTableMeta,
    pub table_type: TableType,
    pub timestamp_column: String,
    pub value_column: String,
}

impl From<TableInfo> for RawTableInfo {
    fn from(info: TableInfo) -> RawTableInfo {
        RawTableInfo {
            ident: info.ident,
            name: info.name,
            desc: info.desc,
            catalog_name: info.catalog_name,
            schema_name: info.schema_name,
            meta: RawTableMeta::from(info.meta),
            table_type: info.table_type,
            timestamp_column: info.timestamp_column,
            value_column: info.value_column,
        }
    }
}

impl TryFrom<RawTableInfo> for TableInfo {
    type Error = ConvertError;

    fn try_from(raw: RawTableInfo) -> ConvertResult<TableInfo> {
        Ok(TableInfo {
            ident: raw.ident,
            name: raw.name,
            desc: raw.desc,
            catalog_name: raw.catalog_name,
            schema_name: raw.schema_name,
            meta: TableMeta::try_from(raw.meta)?,
            table_type: raw.table_type,
            timestamp_column: raw.timestamp_column,
            value_column: raw.value_column,
        })
    }
}

pub fn table_type_to_str(table_type: &TableType) -> &'static str {
    match table_type {
        TableType::Base => "base",
        TableType::View => "view",
        TableType::Temporary => "temporary",
    }
}

pub fn table_type_from_str(table_type: &str) -> Option<TableType> {
    match table_type {
        t if t.eq_ignore_ascii_case("base") => Some(TableType::Base),
        t if t.eq_ignore_ascii_case("view") => Some(TableType::View),
        t if t.eq_ignore_ascii_case("temporary") => Some(TableType::Temporary),
        _ => None,
    }
}

pub fn is_timestamp_compatible(field: DataType) -> bool {
    matches!(
        field,
        DataType::Timestamp(_, _) |
        DataType::Date32 |
        DataType::Date64 |
        DataType::Int64
    )
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, TimeUnit};
    use datafusion::arrow::datatypes::{Schema, SchemaBuilder};

    use crate::table::schema::column_schema::Field;

    use super::*;

    fn new_test_schema() -> Schema {
        let column_schemas = vec![
            Field::new("col1", DataType::Int32, true),
            Field::new("ts", DataType::Time64(TimeUnit::Millisecond), false).with_time_index(true),
            Field::new("col2", DataType::Int32, true),
        ];
        SchemaBuilder::try_from(column_schemas)
            .unwrap()
            .version(123)
            .build()
            .unwrap()
    }

    #[test]
    fn test_raw_convert() {
        let schema = Arc::new(new_test_schema());
        let meta = TableMetaBuilder::default()
            .schema(schema)
            .primary_key_indices(vec![0])
            .engine("query_engine")
            .build()
            .unwrap();
        let info = TableInfoBuilder::default()
            .table_id(10)
            .table_version(5)
            .name("mytable")
            .meta(meta)
            .build()
            .unwrap();

        let raw = RawTableInfo::from(info.clone());
        let info_new = TableInfo::try_from(raw).unwrap();

        assert_eq!(info, info_new);
    }

    #[test]
    fn test_remove_multiple_columns_before_timestamp() {
        let column_schemas = vec![
            Field::new("col1", DataType::Int32, true),
            Field::new("col2", DataType::Int32, true),
            Field::new("col3", DataType::Int32, true),
            Field::new("ts", DataType::Time64(TimeUnit::Millisecond), false).with_time_index(true),
        ];
        let schema = Arc::new(
            SchemaBuilder::try_from(column_schemas)
                .unwrap()
                .version(123)
                .build()
                .unwrap(),
        );
        let meta = TableMetaBuilder::default()
            .schema(schema.clone())
            .primary_key_indices(vec![1])
            .engine("query_engine")
            .build()
            .unwrap();
    }

    #[test]
    fn test_alloc_new_column() {
        let schema = Arc::new(new_test_schema());
        let mut meta = TableMetaBuilder::default()
            .schema(schema)
            .primary_key_indices(vec![0])
            .engine("query_engine")
            .build()
            .unwrap();

        let column_schema = Field::new("col1", DataType::Int32, true);
        let desc = meta.alloc_new_column("test_table", &column_schema).unwrap();

        assert_eq!(column_schema.name(), desc.name);
    }
}
