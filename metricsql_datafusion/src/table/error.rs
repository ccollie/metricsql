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

use datafusion::arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use snafu::Snafu;

use metricsql_common::error::{BoxedError, ErrorExt, StatusCode};

use crate::datatypes::error::DataTypeError;
use crate::table::metadata::TableId;

pub type Result<T> = std::result::Result<T, Error>;
pub type TableResult<T> = std::result::Result<T, Error>;

pub(crate) type QueryError = crate::query::error::Error;

/// Default error implementation of table.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("Datafusion error: {}", source))]
    Datafusion {
        source: ArrowError,
    },

    #[snafu(display("Failed to convert Arrow schema, source: {}", source))]
    SchemaConversion {
        source: ArrowError,
    },

    #[snafu(display("Engine not found: {}", engine))]
    EngineNotFound { engine: String },

    #[snafu(display("Engine exists: {}", engine))]
    EngineExist { engine: String },

    #[snafu(display("Table projection error, source: {}", source))]
    TableProjection {
        source: ArrowError,
    },

    #[snafu(display("Failed to create record batch for Tables, source: {}", source))]
    TablesRecordBatch {
        source: BoxedError,
    },

    #[snafu(display("Column {} already exists in table {}", column_name, table_name))]
    ColumnExists {
        column_name: String,
        table_name: String,
    },

    #[snafu(display("Failed to build schema, msg: {}, source: {}", msg, source))]
    SchemaBuild {
        source: DataTypeError,
        msg: String,
    },

    #[snafu(display("Column {} not exists in table {}", column_name, table_name))]
    ColumnNotExists {
        column_name: String,
        table_name: String,
    },

    #[snafu(display("Duplicated call to plan execute method. table: {}", table))]
    DuplicatedExecuteCall { table: String },

    #[snafu(display("Failed to operate table, source: {}", source))]
    TableOperation { source: BoxedError },

    #[snafu(display("Unsupported operation: {}", operation))]
    Unsupported { operation: String },

    #[snafu(display("Failed to parse table option, key: {}, value: {}", key, value))]
    ParseTableOption {
        key: String,
        value: String,
    },

    #[snafu(display("Invalid table state: {}", table_id))]
    InvalidTable {
        table_id: TableId,
    },

    #[snafu(display("Missing time index column in table: {}", table_name))]
    MissingTimeIndexColumn {
        table_name: String,
    },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        match self {
            Error::Datafusion { .. }
            | Error::SchemaConversion { .. }
            | Error::TableProjection { .. } => StatusCode::EngineExecuteQuery,
            Error::TablesRecordBatch { .. } | Error::DuplicatedExecuteCall { .. } => {
                StatusCode::Unexpected
            }
            Error::ColumnExists { .. } => StatusCode::TableColumnExists,
            Error::SchemaBuild { source, .. } => source.status_code(),
            Error::TableOperation { source } => source.status_code(),
            Error::ColumnNotExists { .. } => StatusCode::TableColumnNotFound,
            Error::Unsupported { .. } => StatusCode::Unsupported,
            Error::ParseTableOption { .. }
            | Error::EngineNotFound { .. }
            | Error::EngineExist { .. } => StatusCode::InvalidArguments,

            Error::InvalidTable { .. } | Error::MissingTimeIndexColumn { .. } => {
                StatusCode::Internal
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl From<Error> for DataFusionError {
    fn from(e: Error) -> DataFusionError {
        DataFusionError::External(Box::new(e))
    }
}