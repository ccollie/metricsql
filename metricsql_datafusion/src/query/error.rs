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

use std::any::Any;
use std::time::Duration;

use arrow_schema::DataType;
use datafusion::error::DataFusionError;
use datafusion_common::ScalarValue;
use snafu::Snafu;

use metricsql_common::error::ext::{BoxedError, ErrorExt};
use metricsql_common::error::status_code::StatusCode;

use crate::datatypes::error::DataTypeError;

pub type DataSourceError = crate::datasource::error::Error;
pub type CatalogError = crate::catalog::error::Error;

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("Unsupported expr type: {}", name))]
    UnsupportedExpr { name: String },

    #[snafu(display("Operation {} not implemented yet", operation))]
    Unimplemented { operation: String },

    #[snafu(display("General catalog error"))]
    Catalog { source: CatalogError },

    #[snafu(display("Catalog not found: {}", catalog))]
    CatalogNotFound { catalog: String },

    #[snafu(display("Schema not found: {}", schema))]
    SchemaNotFound { schema: String },

    #[snafu(display("Table not found: {}", table))]
    TableNotFound { table: String },

    #[snafu(display("Failed to create RecordBatch"))]
    CreateRecordBatch {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("Failed to create Schema"))]
    CreateSchema { source: DataTypeError },

    #[snafu(display("Failure during query execution"))]
    QueryExecution { source: BoxedError },

    #[snafu(display("Failure during query planning"))]
    QueryPlan { source: BoxedError },

    #[snafu(display("Failure during query parsing, query: {}", query))]
    QueryParse { query: String, source: BoxedError },

    #[snafu(display("Illegal access to catalog: {} and schema: {}", catalog, schema))]
    QueryAccessDenied { catalog: String, schema: String },

    #[snafu(display("The SQL string has multiple statements, query: {}", query))]
    MultipleStatements { query: String },

    #[snafu(display("Failed to convert Datafusion schema"))]
    ConvertDatafusionSchema { source: DataTypeError },

    #[snafu(display("Failed to parse timestamp `{}`", raw))]
    ParseTimestamp {
        raw: String,
        #[snafu(source)]
        error: chrono::ParseError,
    },

    #[snafu(display("Failed to parse float number `{}`", raw))]
    ParseFloat {
        raw: String,
        #[snafu(source)]
        error: std::num::ParseFloatError,
    },

    #[snafu(display("DataFusion error"))]
    DataFusion {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Failed to convert DataFusion's recordbatch stream"))]
    ConvertDfRecordBatchStream {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("General SQL error"))]
    Sql { source: crate::sql::error::Error },

    #[snafu(display("Failed to plan SQL"))]
    PlanSql {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Timestamp column for table '{table_name}' is missing!"))]
    MissingTimestampColumn { table_name: String },

    #[snafu(display("Failed to convert value to sql value: {}", value))]
    ConvertSqlValue {
        value: ScalarValue,
        source: crate::sql::error::Error,
    },

    #[snafu(display("Failed to convert concrete type to sql type: {:?}", datatype))]
    ConvertSqlType {
        datatype: DataType,
        source: crate::sql::error::Error,
    },

    #[snafu(display("Missing required field: {}", name))]
    MissingRequiredField { name: String },

    #[snafu(display("Failed to regex"))]
    BuildRegex {
        #[snafu(source)]
        error: regex::Error,
    },

    #[snafu(display("Failed to build data source backend"))]
    BuildBackend { source: DataSourceError },

    #[snafu(display("Failed to list objects"))]
    ListObjects { source: DataSourceError },

    #[snafu(display("Failed to parse file format"))]
    ParseFileFormat { source: DataSourceError },

    #[snafu(display("Failed to infer schema"))]
    InferSchema { source: DataSourceError },

    #[snafu(display("Failed to convert datafusion schema"))]
    ConvertSchema { source: DataTypeError },

    #[snafu(display("Unknown table type, downcast failed"))]
    UnknownTable,

    #[snafu(display("Failed to do vector computation"))]
    VectorComputation { source: DataTypeError },

    #[snafu(display("Cannot find time index column in table {}", table))]
    TimeIndexNotFound { table: String },

    #[snafu(display("Failed to add duration '{:?}' to SystemTime, overflowed", duration))]
    AddSystemTimeOverflow { duration: Duration },

    #[snafu(display(
        "Column schema incompatible, column: {}, file_type: {}, table_type: {}",
        column,
        file_type,
        table_type
    ))]
    ColumnSchemaIncompatible {
        column: String,
        file_type: DataType,
        table_type: DataType,
    },

    #[snafu(display("Column schema has no default value, column: {}", column))]
    ColumnSchemaNoDefault { column: String },

    #[snafu(display("Table mutation error"))]
    TableMutation { source: BoxedError },

    #[snafu(display("Missing table mutation handler"))]
    MissingTableMutationHandler,

    #[snafu(display("Range Query: {}", msg))]
    RangeQuery { msg: String },

    #[snafu(display("Not expected to run ExecutionPlan more than once"))]
    ExecuteRepeatedly,
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;

        match self {
            QueryParse { .. } | MultipleStatements { .. } | RangeQuery { .. } => {
                StatusCode::InvalidSyntax
            }
            UnsupportedExpr { .. }
            | Unimplemented { .. }
            | CatalogNotFound { .. }
            | SchemaNotFound { .. }
            | TableNotFound { .. }
            | UnknownTable { .. }
            | TimeIndexNotFound { .. }
            | ParseTimestamp { .. }
            | ParseFloat { .. }
            | MissingRequiredField { .. }
            | BuildRegex { .. }
            | ConvertSchema { .. }
            | AddSystemTimeOverflow { .. }
            | ColumnSchemaIncompatible { .. }
            | ColumnSchemaNoDefault { .. } => StatusCode::InvalidArguments,

            BuildBackend { .. } | ListObjects { .. } => StatusCode::StorageUnavailable,

            ParseFileFormat { source, .. } | InferSchema { source, .. } => source.status_code(),

            QueryAccessDenied { .. } => StatusCode::AccessDenied,
            Catalog { source, .. } => source.status_code(),
            ConvertDatafusionSchema { source, .. } => source.status_code(),
            CreateRecordBatch { source, .. } => source.status_code(),
            QueryExecution { source, .. } | QueryPlan { source, .. } => source.status_code(),
            DataFusion { error, .. } => match error {
                DataFusionError::Internal(_) => StatusCode::Internal,
                DataFusionError::NotImplemented(_) => StatusCode::Unsupported,
                DataFusionError::Plan(_) => StatusCode::PlanQuery,
                _ => StatusCode::EngineExecuteQuery,
            },
            MissingTimestampColumn { .. } => StatusCode::EngineExecuteQuery,
            Sql { source, .. } => source.status_code(),
            PlanSql { .. } => StatusCode::PlanQuery,
            ConvertSqlType { source, .. } | ConvertSqlValue { source, .. } => source.status_code(),
            CreateSchema { source, .. } => source.status_code(),
            TableMutation { source, .. } => source.status_code(),
            MissingTableMutationHandler { .. } => StatusCode::Unexpected,
            ExecuteRepeatedly { .. } => StatusCode::Unexpected,
            VectorComputation { source, .. } => source.status_code(),
            ConvertDfRecordBatchStream { source, .. } => source.status_code(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for DataFusionError {
    fn from(e: Error) -> DataFusionError {
        DataFusionError::External(Box::new(e))
    }
}
