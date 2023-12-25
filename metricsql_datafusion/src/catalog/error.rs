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
use std::fmt::Debug;

use arrow_schema::DataType;
use datafusion_common::DataFusionError;
use snafu::Snafu;
use tokio::task::JoinError;

use metricsql_common::error::ext::{BoxedError, ErrorExt};
use metricsql_common::error::status_code::StatusCode;

use crate::datatypes::error::DataTypeError;
use crate::table::TableId;

type TableError = crate::table::error::Error;

pub type CatalogError = Error;
pub type CatalogResult<T> = std::result::Result<T, CatalogError>;

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("Failed to list catalogs"))]
    ListCatalogs { source: BoxedError },

    #[snafu(display("Failed to list {}'s schemas", catalog))]
    ListSchemas { catalog: String, source: BoxedError },

    #[snafu(display("Invalid full table name: {}", table_name))]
    InvalidFullTableName { table_name: String },

    #[snafu(display("Failed to open system catalog table"))]
    OpenSystemCatalog { source: TableError },

    #[snafu(display("Failed to create system catalog table"))]
    CreateSystemCatalog { source: TableError },

    #[snafu(display("Failed to create table, table info: {}", table_info))]
    CreateTable {
        table_info: String,
        source: TableError,
    },

    #[snafu(display("System catalog is not valid: {}", msg))]
    SystemCatalog { msg: String },

    #[snafu(display(
        "System catalog table type mismatch, expected: binary, found: {:?}",
        data_type,
    ))]
    SystemCatalogTypeMismatch { data_type: DataType },

    #[snafu(display("Invalid system catalog entry type: {:?}", entry_type))]
    InvalidEntryType { entry_type: Option<u8> },

    #[snafu(display("Invalid system catalog key: {:?}", key))]
    InvalidKey { key: Option<String> },

    #[snafu(display("Catalog value is not present"))]
    EmptyValue,

    #[snafu(display("Failed to deserialize value"))]
    ValueDeserialize {
        #[snafu(source)]
        error: serde_json::error::Error,
    },

    #[snafu(display("Table query_engine not found: {}", engine_name))]
    TableEngineNotFound {
        engine_name: String,
        source: TableError,
    },

    #[snafu(display("Cannot find catalog by name: {}", catalog_name))]
    CatalogNotFound { catalog_name: String },

    #[snafu(display("Cannot find schema {} in catalog {}", schema, catalog))]
    SchemaNotFound { catalog: String, schema: String },

    #[snafu(display("Table `{}` already exists", table))]
    TableExists { table: String },

    #[snafu(display("Table not found: {}", table))]
    TableNotExist { table: String },

    #[snafu(display("Schema {} already exists", schema))]
    SchemaExists { schema: String },

    #[snafu(display("Operation {} not implemented yet", operation))]
    Unimplemented { operation: String },

    #[snafu(display("Operation {} not supported", op))]
    NotSupported { op: String },

    #[snafu(display("Failed to open table {table_id}"))]
    OpenTable {
        table_id: TableId,
        source: TableError,
    },

    #[snafu(display("Failed to open table in parallel"))]
    ParallelOpenTable {
        #[snafu(source)]
        error: JoinError,
    },

    #[snafu(display("Table not found while opening table, table info: {}", table_info))]
    TableNotFound { table_info: String },

    #[snafu(display("Failed to read system catalog table records"))]
    ReadSystemCatalog {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("Failed to create recordbatch"))]
    CreateRecordBatch {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("Failed to insert table creation record to system catalog"))]
    InsertCatalogRecord { source: TableError },

    #[snafu(display("Failed to scan system catalog table"))]
    SystemCatalogTableScan { source: TableError },

    #[snafu(display("Internal error"))]
    Internal { source: BoxedError },

    #[snafu(display("Failed to upgrade weak catalog manager reference"))]
    UpgradeWeakCatalogManagerRef,

    #[snafu(display("Failed to execute system catalog table scan"))]
    SystemCatalogTableScanExec { source: crate::query::error::Error },

    #[snafu(display("Cannot parse catalog value"))]
    InvalidCatalogValue { source: Box<Error> },

    #[snafu(display("Invalid table info in catalog"))]
    InvalidTableInfoInCatalog { source: DataTypeError },

    #[snafu(display("Illegal access to catalog: {} and schema: {}", catalog, schema))]
    QueryAccessDenied { catalog: String, schema: String },

    #[snafu(display("DataFusion error"))]
    Datafusion {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Table schema mismatch"))]
    TableSchemaMismatch { source: TableError },

    #[snafu(display("A generic error has occurred, msg: {}", msg))]
    Generic { msg: String },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        match self {
            Error::InvalidKey { .. }
            | Error::SchemaNotFound { .. }
            | Error::TableNotFound { .. }
            | Error::CatalogNotFound { .. }
            | Error::InvalidEntryType { .. }
            | Error::ParallelOpenTable { .. } => StatusCode::Unexpected,

            Error::SystemCatalog { .. }
            | Error::EmptyValue { .. }
            | Error::ValueDeserialize { .. } => StatusCode::StorageUnavailable,

            Error::Generic { .. }
            | Error::SystemCatalogTypeMismatch { .. }
            | Error::UpgradeWeakCatalogManagerRef { .. } => StatusCode::Internal,

            Error::ReadSystemCatalog { source, .. } | Error::CreateRecordBatch { source, .. } => {
                source.status_code()
            }
            Error::InvalidCatalogValue { source, .. } => source.status_code(),
            Error::InvalidTableInfoInCatalog { source, .. } => source.status_code(),

            Error::TableExists { .. } => StatusCode::TableAlreadyExists,
            Error::TableNotExist { .. } => StatusCode::TableNotFound,
            Error::SchemaExists { .. } | Error::TableEngineNotFound { .. } => {
                StatusCode::InvalidArguments
            }

            Error::ListCatalogs { source, .. } | Error::ListSchemas { source, .. } => {
                source.status_code()
            }

            Error::OpenSystemCatalog { source, .. }
            | Error::CreateSystemCatalog { source, .. }
            | Error::InsertCatalogRecord { source, .. }
            | Error::OpenTable { source, .. }
            | Error::CreateTable { source, .. }
            | Error::TableSchemaMismatch { source, .. } => source.status_code(),

            Error::SystemCatalogTableScan { source, .. } => source.status_code(),
            Error::SystemCatalogTableScanExec { source, .. } => source.status_code(),
            Error::Unimplemented { .. } | Error::NotSupported { .. } => StatusCode::Unsupported,
            Error::QueryAccessDenied { .. } => StatusCode::AccessDenied,
            Error::Datafusion { .. } => StatusCode::EngineExecuteQuery,
            Error::InvalidFullTableName { .. } => {}
            Error::Internal { .. } => StatusCode::Internal,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl From<Error> for DataFusionError {
    fn from(e: Error) -> Self {
        DataFusionError::Internal(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
