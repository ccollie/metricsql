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
use serde_json::error::Error as JsonError;
use snafu::Snafu;

use metricsql_common::prelude::{ErrorExt, status_code::StatusCode};

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("Unsupported operation: {}", operation))]
    Unsupported { operation: String },

    #[snafu(display("Unexpected query_engine: {}", engine))]
    UnexpectedEngine { engine: String },

    #[snafu(display("Failed to check object from path: {}", path))]
    CheckObject {
        path: String,

        #[snafu(source)]
        error: crate::object_store::Error,
    },

    #[snafu(display("Fail to encode object into json"))]
    EncodeJson {
        #[snafu(source)]
        error: JsonError,
    },

    #[snafu(display("Fail to decode object from json"))]
    DecodeJson {
        #[snafu(source)]
        error: JsonError,
    },

    #[snafu(display("Missing required field: {}", name))]
    MissingRequiredField { name: String },

    #[snafu(display("Failed to build backend"))]
    BuildBackend {
        source: crate::datasource::error::Error,
    },

    #[snafu(display("Failed to build csv config"))]
    BuildCsvConfig {
        #[snafu(source)]
        error: crate::datasource::file_format::csv::CsvConfigBuilderError,
    },

    #[snafu(display("Failed to build stream"))]
    BuildStream {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Failed to project schema"))]
    ProjectArrowSchema {
        #[snafu(source)]
        error: ArrowError,
    },

    #[snafu(display("Failed to project schema"))]
    ProjectSchema {
        #[snafu(source)]
        error: ArrowError,
    },

    #[snafu(display("Failed to build stream adapter"))]
    BuildStreamAdapter {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("Failed to parse file format"))]
    ParseFileFormat {
        #[snafu(source)]
        source: crate::datasource::error::Error,
    },

    #[snafu(display("Failed to generate parquet scan plan"))]
    ParquetScanPlan {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display(
        "Projection out of bounds, column_index: {}, bounds: {}",
        column_index,
        bounds
    ))]
    ProjectionOutOfBounds { column_index: usize, bounds: usize },

    #[snafu(display("Failed to extract column from filter"))]
    ExtractColumnFromFilter {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Failed to create default value for column: {}", column))]
    CreateDefault {
        column: String,
        source: ArrowError,
    },

    #[snafu(display("Missing default value for column: {}", column))]
    MissingColumnNoDefault { column: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;

        match self {
            BuildCsvConfig { .. }
            | ProjectArrowSchema { .. }
            | ProjectSchema { .. }
            | MissingRequiredField { .. }
            | Unsupported { .. }
            | ProjectionOutOfBounds { .. }
            | CreateDefault { .. }
            | MissingColumnNoDefault { .. } => StatusCode::InvalidArguments,

            BuildBackend { source, .. } => source.status_code(),
            BuildStreamAdapter { source, .. } => source.status_code(),
            ParseFileFormat { source, .. } => source.status_code(),

            CheckObject { .. } => StatusCode::StorageUnavailable,

            EncodeJson { .. }
            | DecodeJson { .. }
            | BuildStream { .. }
            | ParquetScanPlan { .. }
            | UnexpectedEngine { .. }
            | ExtractColumnFromFilter { .. } => StatusCode::Unexpected,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
