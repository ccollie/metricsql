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

use metricsql_common::error::ext::ErrorExt;
use metricsql_common::error::status_code::StatusCode;
use snafu::Snafu;
use std::any::Any;
use std::fmt::Debug;

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("Failed to serialize data"))]
    Serialize {
        #[snafu(source)]
        error: serde_json::Error,
    },

    #[snafu(display("Failed to deserialize data, json: {}", json))]
    Deserialize {
        #[snafu(source)]
        error: serde_json::Error,
        json: String,
    },

    #[snafu(display("Failed to convert datafusion type: {}", from))]
    Conversion { from: String },

    #[snafu(display("Bad array access, Index out of bounds: {}, size: {}", index, size))]
    BadArrayAccess { index: usize, size: usize },

    #[snafu(display("Unsupported arrow data type, type: {:?}", arrow_type))]
    UnsupportedArrowType {
        arrow_type: arrow::datatypes::DataType,
    },

    #[snafu(display("Unsupported operation: {} for vector: {}", op, vector_type))]
    UnsupportedOperation { op: String, vector_type: String },

    #[snafu(display("Failed to parse version in schema meta, value: {}", value))]
    ParseSchemaVersion {
        value: String,
        #[snafu(source)]
        error: std::num::ParseIntError,
    },

    #[snafu(display("Invalid timestamp index: {}", index))]
    InvalidTimestampIndex { index: usize },

    #[snafu(display("{}", msg))]
    CastType { msg: String },

    #[snafu(display("Failed to cast arrow time i32 type into i64"))]
    CastTimeType {
        #[snafu(source)]
        error: std::num::TryFromIntError,
    },

    #[snafu(display("Arrow failed to compute"))]
    ArrowCompute {
        #[snafu(source)]
        error: arrow::error::ArrowError,
    },

    #[snafu(display("Failed to project arrow schema"))]
    ProjectArrowSchema {
        #[snafu(source)]
        error: arrow::error::ArrowError,
    },

    #[snafu(display("Default value should not be null for non null column"))]
    NullDefault,

    #[snafu(display("Incompatible default value type, reason: {}", reason))]
    DefaultValueType { reason: String },

    #[snafu(display("Duplicated metadata for {}", key))]
    DuplicateMeta { key: String },

    #[snafu(display("Failed to convert value into scalar value, reason: {}", reason))]
    ToScalarValue { reason: String },

    #[snafu(display("Invalid timestamp precision: {}", precision))]
    InvalidTimestampPrecision { precision: u64 },

    #[snafu(display("Column {} already exists", column))]
    DuplicateColumn { column: String },

    #[snafu(display("Failed to unpack value to given type: {}", reason))]
    TryFromValue { reason: String },

    #[snafu(display("Failed to specify the precision {} and scale {}", precision, scale))]
    InvalidPrecisionOrScale {
        precision: u8,
        scale: i8,
        #[snafu(source)]
        error: arrow::error::ArrowError,
    },

    #[snafu(display("Value exceeds the precision {} bound", precision))]
    ValueExceedsPrecision {
        precision: u8,
        #[snafu(source)]
        error: arrow::error::ArrowError,
    },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        // Inner encoding and decoding error should not be exposed to users.
        StatusCode::Internal
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;
pub type DataTypeError = Error;
