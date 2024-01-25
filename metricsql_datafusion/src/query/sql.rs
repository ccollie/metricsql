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

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::compute::can_cast_types;
use arrow_schema::{DataType, Field, FieldRef, Schema, TimeUnit};
use datafusion_common::ScalarValue;
use once_cell::sync::Lazy;
use regex::Regex;
use snafu::{ensure, OptionExt, ResultExt};

use crate::catalog::CatalogManagerRef;
use crate::catalog::consts::{SEMANTIC_TYPE_FIELD, SEMANTIC_TYPE_PRIMARY_KEY, SEMANTIC_TYPE_TIME_INDEX};
use crate::common::recordbatch::{RecordBatch, RecordBatches};
use crate::datasource::file_format::{FileFormat, Format, infer_schemas};
use crate::datasource::lister::{Lister, Source};
use crate::datasource::object_store::build_backend;
use crate::datasource::util::find_dir_and_filename;
use crate::datatypes::{data_type_name, table_type_name};
use crate::object_store::ObjectStore;
use crate::query::datafusion::execute_show_with_filter;
use crate::query::error::{
    BuildBackendSnafu, BuildRegexSnafu, CatalogSnafu, ColumnSchemaIncompatibleSnafu,
    ColumnSchemaNoDefaultSnafu, CreateRecordBatchSnafu, InferSchemaSnafu, ListObjectsSnafu,
    MissingRequiredFieldSnafu, ParseFileFormatSnafu, Result, VectorComputationSnafu
};
use crate::query::helper::Helper;
use crate::query::Output;
use crate::session::context::QueryContextRef;
use crate::sql::statements::show::{ShowDatabases, ShowKind, ShowTables};
use crate::table::requests::{FILE_TABLE_LOCATION_KEY, FILE_TABLE_PATTERN_KEY};
use crate::table::schema::raw::RawSchema;
use crate::table::{is_timestamp_field, TableRef};

const SCHEMAS_COLUMN: &str = "Schemas";
const TABLES_COLUMN: &str = "Tables";
const COLUMN_NAME_COLUMN: &str = "Column";
const COLUMN_TYPE_COLUMN: &str = "Type";
const COLUMN_KEY_COLUMN: &str = "Key";
const COLUMN_NULLABLE_COLUMN: &str = "Null";
const COLUMN_DEFAULT_COLUMN: &str = "Default";
const COLUMN_SEMANTIC_TYPE_COLUMN: &str = "Semantic Type";

const NULLABLE_YES: &str = "YES";
const NULLABLE_NO: &str = "NO";
const PRI_KEY: &str = "PRI";

const GREPTIME_TIMESTAMP: &str = "greptime_timestamp";

static DESCRIBE_TABLE_OUTPUT_SCHEMA: Lazy<Arc<Schema>> = Lazy::new(|| {
    Arc::new(Schema::new(vec![
        Field::new(
            COLUMN_NAME_COLUMN,
            DataType::Utf8,
            false,
        ),
        Field::new(
            COLUMN_TYPE_COLUMN,
            DataType::Utf8,
            false,
        ),
        Field::new(COLUMN_KEY_COLUMN, DataType::Utf8, true),
        Field::new(
            COLUMN_NULLABLE_COLUMN,
            DataType::Utf8,
            false,
        ),
        Field::new(
            COLUMN_DEFAULT_COLUMN,
            DataType::Utf8,
            false,
        ),
        Field::new(
            COLUMN_SEMANTIC_TYPE_COLUMN,
            DataType::Utf8,
            false,
        ),
    ]))
});

static SHOW_CREATE_TABLE_OUTPUT_SCHEMA: Lazy<Arc<Schema>> = Lazy::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("Table", DataType::Utf8, false),
        Field::new("Create Table", DataType::Utf8, false),
    ]))
});

pub async fn show_databases(
    stmt: ShowDatabases,
    catalog_manager: CatalogManagerRef,
    query_ctx: QueryContextRef,
) -> Result<Output> {
    let mut databases = catalog_manager
        .schema_names(query_ctx.current_catalog())
        .await
        .context(CatalogSnafu)?;

    // TODO(dennis): Specify the order of the results in catalog manager API
    databases.sort();

    let schema = Arc::new(Schema::new(vec![Field::new(
        SCHEMAS_COLUMN,
        DataType::Utf8,
        false,
    )]));
    match stmt.kind {
        ShowKind::All => {
            let databases = Arc::new(StringArray::from(databases)) as _;
            let records = RecordBatches::try_from_columns(schema, vec![databases])
                .context(CreateRecordBatchSnafu)?;
            Ok(Output::RecordBatches(records))
        }
        ShowKind::Where(filter) => {
            let columns = vec![Arc::new(StringArray::from(databases)) as _];
            let record_batch =
                RecordBatch::new(schema, columns).context(CreateRecordBatchSnafu)?;
            let result = execute_show_with_filter(record_batch, Some(filter)).await?;
            Ok(result)
        }
        ShowKind::Like(ident) => {
            let databases = Helper::like_utf8(databases, &ident)
                .context(VectorComputationSnafu)?;
            let records = RecordBatches::try_from_columns(schema, vec![databases])
                .context(CreateRecordBatchSnafu)?;
            Ok(Output::RecordBatches(records))
        }
    }
}

pub async fn show_tables(
    stmt: ShowTables,
    catalog_manager: CatalogManagerRef,
    query_ctx: QueryContextRef,
) -> Result<Output> {
    let schema_name = if let Some(database) = stmt.database {
        database
    } else {
        query_ctx.current_schema().to_owned()
    };
    // TODO(sunng87): move this function into query_ctx
    let mut tables = catalog_manager
        .table_names(query_ctx.current_catalog(), &schema_name)
        .await
        .context(CatalogSnafu)?;

    // TODO(dennis): Specify the order of the results in schema provider API
    tables.sort();

    let table_types: Option<Arc<dyn Array>> = {
        if stmt.full {
            Some(
                get_table_types(
                    &tables,
                    catalog_manager.clone(),
                    query_ctx.clone(),
                    &schema_name,
                )
                .await?,
            )
        } else {
            None
        }
    };

    let mut column_schema = vec![Field::new(
        TABLES_COLUMN,
        DataType::Utf8,
        false,
    )];
    if table_types.is_some() {
        column_schema.push(Field::new(
            "Table_type",
            DataType::Utf8,
            false,
        ));
    }

    let schema = Arc::new(Schema::new(column_schema));

    match stmt.kind {
        ShowKind::All => {
            let tables = Arc::new(StringArray::from(tables)) as _;
            let mut columns = vec![tables];
            if let Some(table_types) = table_types {
                columns.push(table_types)
            }

            let records = RecordBatches::try_from_columns(schema, columns)
                .context(CreateRecordBatchSnafu)?;
            Ok(Output::RecordBatches(records))
        }
        ShowKind::Where(filter) => {
            let mut columns = vec![Arc::new(StringArray::from(tables)) as _];
            if let Some(table_types) = table_types {
                columns.push(table_types)
            }
            let record_batch =
                RecordBatch::new(schema, columns).context(CreateRecordBatchSnafu)?;
            let result = execute_show_with_filter(record_batch, Some(filter)).await?;
            Ok(result)
        }
        ShowKind::Like(ident) => {
            let (tables, filter) = Helper::like_utf8_filter(tables, &ident)
                .context(VectorComputationSnafu)?;
            let mut columns = vec![tables];

            if let Some(table_types) = table_types {
                let table_types = table_types
                    .filter(&filter)
                    .context(VectorComputationSnafu)?;
                columns.push(table_types)
            }

            let records = RecordBatches::try_from_columns(schema, columns)
                .context(CreateRecordBatchSnafu)?;
            Ok(Output::RecordBatches(records))
        }
    }
}

pub fn describe_table(table: TableRef) -> Result<Output> {
    let table_info = table.table_info();
    let columns_schemas = table_info.meta.schema.fields().as_ref();
    let columns = vec![
        describe_column_names(columns_schemas),
        describe_column_types(columns_schemas),
        describe_column_keys(columns_schemas, &table_info.meta.primary_key_indices),
        describe_column_nullables(columns_schemas),
        describe_column_semantic_types(columns_schemas, &table_info.meta.primary_key_indices),
    ];
    let records = RecordBatches::try_from_columns(DESCRIBE_TABLE_OUTPUT_SCHEMA.clone(), columns)
        .context(CreateRecordBatchSnafu)?;
    Ok(Output::RecordBatches(records))
}

fn describe_column_names(columns_schemas: &[FieldRef]) -> ArrayRef {
    Arc::new(StringArray::from(
        columns_schemas.iter().map(|cs| cs.name().as_str()).collect::<Vec<_>>()
    ))
}

fn describe_column_types(columns_schemas: &[FieldRef]) -> ArrayRef {
    Arc::new(StringArray::from(
        columns_schemas
            .iter()
            .map(|cs| data_type_name(cs.data_type()))
            .collect::<Vec<_>>(),
    ))
}

fn describe_column_keys(
    columns_schemas: &[FieldRef],
    primary_key_indices: &[usize],
) -> ArrayRef {
    let vals = columns_schemas.iter().enumerate().map(|(i, cs)| {
        if is_timestamp_field(cs) || primary_key_indices.contains(&i) {
            PRI_KEY
        } else {
            ""
        }
    }).collect::<Vec<_>>();

    Arc::new(StringArray::from(vals))
}

fn describe_column_nullables(columns_schemas: &[FieldRef]) -> ArrayRef {
    Arc::new(StringArray::from(columns_schemas.iter().map(
        |cs| {
            if cs.is_nullable() {
                NULLABLE_YES
            } else {
                NULLABLE_NO
            }
        },
    ).collect::<Vec<_>>()))
}

fn describe_column_semantic_types(
    columns_schemas: &[FieldRef],
    primary_key_indices: &[usize],
) -> ArrayRef {
    Arc::new(StringArray::from(
        columns_schemas.iter().enumerate().map(|(i, cs)| {
            if primary_key_indices.contains(&i) {
                SEMANTIC_TYPE_PRIMARY_KEY
            } else if is_timestamp_field(cs) {
                SEMANTIC_TYPE_TIME_INDEX
            } else {
                SEMANTIC_TYPE_FIELD
            }
        }).collect::<Vec<_>>()
    ))
}

// lists files in the frontend to reduce unnecessary scan requests repeated in each datanode.
pub async fn prepare_file_table_files(
    options: &HashMap<String, String>,
) -> Result<(ObjectStore, Vec<String>)> {
    let url = options
        .get(FILE_TABLE_LOCATION_KEY)
        .context(MissingRequiredFieldSnafu {
            name: FILE_TABLE_LOCATION_KEY,
        })?;

    let (dir, filename) = find_dir_and_filename(url);
    let source = if let Some(filename) = filename {
        Source::Filename(filename)
    } else {
        Source::Dir
    };
    let regex = options
        .get(FILE_TABLE_PATTERN_KEY)
        .map(|x| Regex::new(x))
        .transpose()
        .context(BuildRegexSnafu)?;
    let object_store = build_backend(url, options).context(BuildBackendSnafu)?;
    let lister = Lister::new(object_store.clone(), source, dir, regex);
    // If we scan files in a directory every time the database restarts,
    // then it might lead to a potential undefined behavior:
    // If a user adds a file with an incompatible schema to that directory,
    // it will make the external table unavailable.
    let files = lister
        .list()
        .await
        .context(ListObjectsSnafu)?
        .into_iter()
        .filter_map(|entry| {
            if entry.path().ends_with('/') {
                None
            } else {
                Some(entry.path().to_string())
            }
        })
        .collect::<Vec<_>>();
    Ok((object_store, files))
}

pub async fn infer_file_table_schema(
    object_store: &ObjectStore,
    files: &[String],
    options: &HashMap<String, String>,
) -> Result<RawSchema> {
    let format = parse_file_table_format(options)?;
    let merged = infer_schemas(object_store, files, format.as_ref())
        .await
        .context(InferSchemaSnafu)?;
    Ok(RawSchema::from(&merged))
}

// Converts the file column schemas to table column schemas.
// Returns the column schemas and the time index column name.
//
// More specifically, this function will do the following:
// 1. Add a default time index column if there is no time index column
//    in the file column schemas, or
// 2. If the file column schemas contain a column with name conflicts with
//    the default time index column, it will replace the column schema
//    with the default one.
pub fn file_column_schemas_to_table(
    file_column_schemas: &[Field],
) -> (Vec<Field>, String) {
    let mut column_schemas = file_column_schemas.to_owned();
    if let Some(time_index_column) = column_schemas.iter().find(|c| is_timestamp_field(c)) {
        let time_index = time_index_column.name().clone();
        return (column_schemas, time_index);
    }

    let timestamp_type = DataType::Timestamp(TimeUnit::Millisecond, None);
    let default_zero = ScalarValue::TimestampMillisecond(Some(0), None);
    let timestamp_column_schema = Field::new(GREPTIME_TIMESTAMP, timestamp_type, false)
        .with_time_index(true)
       // .with_default_constraint(Some(ColumnDefaultConstraint::Value(default_zero)))
        .unwrap();

    if let Some(column_schema) = column_schemas
        .iter_mut()
        .find(|column_schema| column_schema.name() == GREPTIME_TIMESTAMP)
    {
        // Replace the column schema with the default one
        *column_schema = timestamp_column_schema;
    } else {
        column_schemas.push(timestamp_column_schema);
    }

    (column_schemas, GREPTIME_TIMESTAMP.to_string())
}

/// This function checks if the column schemas from a file can be matched with
/// the column schemas of a table.
///
/// More specifically, for each column seen in the table schema,
/// - If the same column does exist in the file schema, it checks if the data
/// type of the file column can be casted into the form of the table column.
/// - If the same column does not exist in the file schema, it checks if the
/// table column is nullable or has a default constraint.
pub fn check_file_to_table_schema_compatibility(
    file_column_schemas: &[Field],
    table_column_schemas: &[Field],
) -> Result<()> {
    let file_schemas_map = file_column_schemas
        .iter()
        .map(|s| (s.name().clone(), s))
        .collect::<HashMap<_, _>>();

    for table_column in table_column_schemas {
        let field_name = table_column.name();
        if let Some(file_column) = file_schemas_map.get(field_name) {
            // TODO(zhongzc): a temporary solution, we should use `can_cast_to` once it's ready.
            ensure!(
                can_cast_types(file_column.data_type(), table_column.data_type()),
                ColumnSchemaIncompatibleSnafu {
                    column: table_column.name().clone(),
                    file_type: file_column.data_type().clone(),
                    table_type: table_column.data_type().clone(),
                }
            );
        } else {
            ensure!(
                table_column.is_nullable(),
                ColumnSchemaNoDefaultSnafu {
                    column: table_column.name().clone(),
                }
            );
        }
    }

    Ok(())
}

fn parse_file_table_format(options: &HashMap<String, String>) -> Result<Box<dyn FileFormat>> {
    Ok(
        match Format::try_from(options).context(ParseFileFormatSnafu)? {
            Format::Csv(format) => Box::new(format),
            Format::Json(format) => Box::new(format),
            Format::Parquet(format) => Box::new(format),
            Format::Orc(format) => Box::new(format),
        },
    )
}

async fn get_table_types(
    tables: &[String],
    catalog_manager: CatalogManagerRef,
    query_ctx: QueryContextRef,
    schema_name: &str,
) -> Result<Arc<dyn Array>> {
    let mut table_types = Vec::with_capacity(tables.len());
    for table_name in tables {
        if let Some(table) = catalog_manager
            .table(query_ctx.current_catalog(), schema_name, table_name)
            .await
            .context(CatalogSnafu)?
        {
            table_types.push(table_type_name(table.table_type()).to_string());
        }
    }
    Ok(Arc::new(StringArray::from(table_types)) as _)
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use arrow::array::{ArrayRef, StringArray, TimestampMillisecondArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
    use datafusion::datasource::MemTable;
    use snafu::ResultExt;

    use crate::catalog::consts::{SEMANTIC_TYPE_FIELD, SEMANTIC_TYPE_TIME_INDEX};
    use crate::catalog::error::CreateRecordBatchSnafu;
    use crate::common::recordbatch::{RecordBatch, RecordBatches};
    use crate::error::Result;
    use crate::query::Output;
    use crate::query::sql::{describe_table, DESCRIBE_TABLE_OUTPUT_SCHEMA, NULLABLE_NO, NULLABLE_YES};
    use crate::table::schema::column_schema::Field;
    use crate::table::TableRef;

    #[test]
    fn test_describe_table_multiple_columns() -> Result<()> {
        let table_name = "test_table";
        let schema = vec![
            Field::new("t1", DataType::UInt32, true),
            Field::new(
                "t2",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            )
            .unwrap()
            .with_time_index(true),
        ];
        let data = vec![
            Arc::new(UInt32Array::from_slice([0])) as _,
            Arc::new(TimestampMillisecondArray::from_slice([0])) as _,
        ];
        let expected_columns = vec![
            Arc::new(StringArray::from(vec!["t1", "t2"])) as _,
            Arc::new(StringArray::from(vec!["UInt32", "TimestampMillisecond"])) as _,
            Arc::new(StringArray::from(vec!["", "PRI"])) as _,
            Arc::new(StringArray::from(vec![NULLABLE_YES, NULLABLE_NO])) as _,
            Arc::new(StringArray::from(vec!["", "current_timestamp()"])) as _,
            Arc::new(StringArray::from(vec![
                SEMANTIC_TYPE_FIELD,
                SEMANTIC_TYPE_TIME_INDEX,
            ])) as _,
        ];

        describe_table_test_by_schema(table_name, schema, data, expected_columns)
    }

    fn describe_table_test_by_schema(
        table_name: &str,
        schema: Vec<Field>,
        data: Vec<ArrayRef>,
        expected_columns: Vec<ArrayRef>,
    ) -> Result<()> {
        let table_schema = SchemaRef::new(Schema::new(schema));
        let table = prepare_describe_table(table_name, table_schema, data);

        let expected =
            RecordBatches::try_from_columns(DESCRIBE_TABLE_OUTPUT_SCHEMA.clone(), expected_columns)
                .context(CreateRecordBatchSnafu)?;

        if let Output::RecordBatches(res) = describe_table(table)? {
            assert_eq!(res.take(), expected.take());
        } else {
            panic!("describe table must return record batch");
        }

        Ok(())
    }

    fn prepare_describe_table(
        table_name: &str,
        table_schema: SchemaRef,
        data: Vec<ArrayRef>,
    ) -> TableRef {
        let record_batch = RecordBatch::new(table_schema, data).unwrap();
        MemTable::table(table_name, record_batch)
    }
}
