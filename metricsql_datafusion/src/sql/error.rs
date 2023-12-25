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
use datafusion_common::ScalarValue;
use snafu::Snafu;

use metricsql_common::prelude::{ErrorExt, StatusCode};

use crate::datatypes::error::DataTypeError;

pub type Result<T> = std::result::Result<T, Error>;

/// SQL parser errors.
// Now the error in parser does not contains backtrace to avoid generating backtrace
// every time the parser parses an invalid SQL.
#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("SQL statement is not supported: {}, keyword: {}", sql, keyword))]
    Unsupported { sql: String, keyword: String },

    #[snafu(display("Missing time index constraint"))]
    MissingTimeIndex {},

    #[snafu(display("Invalid time index: {}", msg))]
    InvalidTimeIndex { msg: String },

    #[snafu(display("Invalid SQL, error: {}", msg))]
    InvalidSql { msg: String },

    #[snafu(display("Invalid column option, column name: {}, error: {}", name, msg))]
    InvalidColumnOption { name: String, msg: String },

    #[snafu(display("SQL data type not supported yet: {:?}", t))]
    SqlTypeNotSupported { t: DataType },

    #[snafu(display("Failed to parse value: {}", msg))]
    ParseSqlValue { msg: String },

    #[snafu(display(
        "Column {} expect type: {:?}, actual: {:?}",
        column_name,
        expect,
        actual,
    ))]
    ColumnTypeMismatch {
        column_name: String,
        expect: DataType,
        actual: DataType,
    },

    #[snafu(display("Invalid database name: {}", name))]
    InvalidDatabaseName { name: String },

    #[snafu(display("Invalid table name: {}", name))]
    InvalidTableName { name: String },

    #[snafu(display("Invalid default constraint, column: {}", column))]
    InvalidDefault {
        column: String,
        source: DataTypeError,
    },

    #[snafu(display("Failed to cast SQL value {} to datatype {}", sql_value, datatype))]
    InvalidCast {
        sql_value: ScalarValue,
        datatype: DataType,
        source: DataTypeError,
    },

    #[snafu(display("Invalid table option key: {}", key))]
    InvalidTableOption { key: String },

    #[snafu(display("Failed to serialize column default constraint"))]
    SerializeColumnDefaultConstraint {
        source: DataTypeError,
    },

    #[snafu(display("Invalid sql value: {}", value))]
    InvalidSqlValue { value: String },

    #[snafu(display("Unable to convert statement {} to DataFusion statement", statement))]
    ConvertToDfStatement {
        statement: String,
    },

    #[snafu(display("Unable to convert sql value {} to datatype {:?}", value, datatype))]
    ConvertSqlValue {
        value: ScalarValue,
        datatype: DataType,
    },

    #[snafu(display("Unable to convert value {} to sql value", value))]
    ConvertValue { value: ScalarValue },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;

        match self {
            Unsupported { .. } => StatusCode::Unsupported,
            MissingTimeIndex { .. }
            | InvalidTimeIndex { .. }
            | InvalidSql { .. }
            | ParseSqlValue { .. }
            | SqlTypeNotSupported { .. }
            | InvalidDefault { .. } => StatusCode::InvalidSyntax,

            InvalidColumnOption { .. }
            | InvalidDatabaseName { .. }
            | ColumnTypeMismatch { .. }
            | InvalidTableName { .. }
            | InvalidSqlValue { .. }
            | InvalidTableOption { .. }
            | InvalidCast { .. } => StatusCode::InvalidArguments,

            SerializeColumnDefaultConstraint { source, .. } => source.status_code(),
            ConvertToDfStatement { .. } => StatusCode::Internal,
            ConvertSqlValue { .. } | ConvertValue { .. } => StatusCode::Unsupported,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
