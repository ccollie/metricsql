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

use datafusion_common::DataFusionError;
use snafu::{Location, Snafu};
use tokio::task::JoinError;

use crate::common::{BoxedError, ErrorExt};
use crate::datatypes::data_type::ConcreteDataType;
use crate::status_code::StatusCode;
use crate::table::metadata::TableId;

type TableError = crate::table::error::Error;

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[stack_trace_debug]
pub enum Error {
    #[snafu(display("Failed to list catalogs"))]
    ListCatalogs {
        location: Location,
        source: BoxedError,
    },

    #[snafu(display("Failed to list {}'s schemas", catalog))]
    ListSchemas {
        location: Location,
        catalog: String,
        source: BoxedError,
    },

    #[snafu(display("Failed to re-compile script due to internal error"))]
    CompileScriptInternal {
        location: Location,
        source: BoxedError,
    },
    #[snafu(display("Failed to open system catalog table"))]
    OpenSystemCatalog {
        location: Location,
        source: TableError,
    },

    #[snafu(display("Failed to create system catalog table"))]
    CreateSystemCatalog {
        location: Location,
        source: TableError,
    },

    #[snafu(display("Failed to create table, table info: {}", table_info))]
    CreateTable {
        table_info: String,
        location: Location,
        source: TableError,
    },

    #[snafu(display("System catalog is not valid: {}", msg))]
    SystemCatalog { msg: String, location: Location },

    #[snafu(display(
        "System catalog table type mismatch, expected: binary, found: {:?}",
        data_type,
    ))]
    SystemCatalogTypeMismatch {
        data_type: ConcreteDataType,
        location: Location,
    },

    #[snafu(display("Invalid system catalog entry type: {:?}", entry_type))]
    InvalidEntryType {
        entry_type: Option<u8>,
        location: Location,
    },

    #[snafu(display("Invalid system catalog key: {:?}", key))]
    InvalidKey {
        key: Option<String>,
        location: Location,
    },

    #[snafu(display("Catalog value is not present"))]
    EmptyValue { location: Location },

    #[snafu(display("Failed to deserialize value"))]
    ValueDeserialize {
        #[snafu(source)]
        error: serde_json::error::Error,
        location: Location,
    },

    #[snafu(display("Table engine not found: {}", engine_name))]
    TableEngineNotFound {
        engine_name: String,
        location: Location,
        source: TableError,
    },

    #[snafu(display("Cannot find catalog by name: {}", catalog_name))]
    CatalogNotFound {
        catalog_name: String,
        location: Location,
    },

    #[snafu(display("Cannot find schema {} in catalog {}", schema, catalog))]
    SchemaNotFound {
        catalog: String,
        schema: String,
        location: Location,
    },

    #[snafu(display("Table `{}` already exists", table))]
    TableExists { table: String, location: Location },

    #[snafu(display("Table not found: {}", table))]
    TableNotExist { table: String, location: Location },

    #[snafu(display("Schema {} already exists", schema))]
    SchemaExists { schema: String, location: Location },

    #[snafu(display("Operation {} not implemented yet", operation))]
    Unimplemented {
        operation: String,
        location: Location,
    },

    #[snafu(display("Operation {} not supported", op))]
    NotSupported { op: String, location: Location },

    #[snafu(display("Failed to open table {table_id}"))]
    OpenTable {
        table_id: TableId,
        location: Location,
        source: TableError,
    },

    #[snafu(display("Failed to open table in parallel"))]
    ParallelOpenTable {
        #[snafu(source)]
        error: JoinError,
    },

    #[snafu(display("Table not found while opening table, table info: {}", table_info))]
    TableNotFound {
        table_info: String,
        location: Location,
    },

    #[snafu(display("Failed to read system catalog table records"))]
    ReadSystemCatalog {
        location: Location,
        source: common_recordbatch::error::Error,
    },

    #[snafu(display("Failed to create recordbatch"))]
    CreateRecordBatch {
        location: Location,
        source: common_recordbatch::error::Error,
    },

    #[snafu(display("Failed to insert table creation record to system catalog"))]
    InsertCatalogRecord {
        location: Location,
        source: TableError,
    },

    #[snafu(display("Failed to scan system catalog table"))]
    SystemCatalogTableScan {
        location: Location,
        source: TableError,
    },

    #[snafu(display("Internal error"))]
    Internal {
        location: Location,
        source: BoxedError,
    },

    #[snafu(display("Failed to upgrade weak catalog manager reference"))]
    UpgradeWeakCatalogManagerRef { location: Location },

    #[snafu(display("Failed to execute system catalog table scan"))]
    SystemCatalogTableScanExec {
        location: Location,
        source: common_query::error::Error,
    },

    #[snafu(display("Cannot parse catalog value"))]
    InvalidCatalogValue {
        location: Location,
        source: common_catalog::error::Error,
    },

    #[snafu(display("Invalid table info in catalog"))]
    InvalidTableInfoInCatalog {
        location: Location,
        source: crate::datatypes::error::Error,
    },

    #[snafu(display("Illegal access to catalog: {} and schema: {}", catalog, schema))]
    QueryAccessDenied { catalog: String, schema: String },

    #[snafu(display("DataFusion error"))]
    Datafusion {
        #[snafu(source)]
        error: DataFusionError,
        location: Location,
    },

    #[snafu(display("Table schema mismatch"))]
    TableSchemaMismatch {
        location: Location,
        source: TableError,
    },

    #[snafu(display("A generic error has occurred, msg: {}", msg))]
    Generic { msg: String, location: Location },

    #[snafu(display("Table metadata manager error"))]
    TableMetadataManager {
        source: common_meta::error::Error,
        location: Location,
    },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        match self {
            Error::InvalidCatalog { .. }
            | Error::DeserializeCatalogEntryValue { .. }
            | Error::SerializeCatalogEntryValue { .. } => StatusCode::Unexpected,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;
