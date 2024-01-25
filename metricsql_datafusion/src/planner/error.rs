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

use datafusion::error::DataFusionError;
use snafu::Snafu;

use metricsql_common::error::ext::ErrorExt;
use metricsql_common::error::status_code::StatusCode;
use metricsql_parser::ast::VectorMatchCardinality;

#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum Error {
    #[snafu(display("Unsupported expr type: {}", name))]
    UnsupportedExpr { name: String },

    #[snafu(display("Unsupported vector matches: {:?}", name))]
    UnsupportedVectorMatch { name: VectorMatchCardinality },

    #[snafu(display("Internal error during building DataFusion plan"))]
    DataFusionPlanning {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("Unexpected plan or expression: {}", desc))]
    UnexpectedPlanExpr { desc: String },

    #[snafu(display("Unknown table type, downcast failed"))]
    UnknownTable,

    #[snafu(display("Cannot find time index column in table {}", table))]
    TimeIndexNotFound { table: String },

    #[snafu(display("Cannot find value columns in table {}", table))]
    ValueNotFound { table: String },

    #[snafu(display(
        "Illegal range: offset {}, length {}, array len {}",
        offset,
        length,
        len,
    ))]
    IllegalRange {
        offset: u32,
        length: u32,
        len: usize,
    },

    #[snafu(display("Failed to deserialize"))]
    Deserialize {
        #[snafu(source)]
        error: prost::DecodeError,
    },

    #[snafu(display("Empty range is not expected"))]
    EmptyRange,

    #[snafu(display(
        "Table (metric) name not found, this indicates a procedure error in PromQL planner"
    ))]
    TableNameNotFound,

    #[snafu(display("General catalog error: "))]
    Catalog {
        source: crate::catalog::error::Error,
    },

    #[snafu(display("Expect a range selector, but not found"))]
    ExpectRangeSelector,

    #[snafu(display("Zero range in range selector"))]
    ZeroRangeSelector,

    #[snafu(display("Cannot find column {col}"))]
    ColumnNotFound { col: String },

    #[snafu(display("Found multiple metric matchers ({count}) in selector"))]
    MultipleMetricMatchers { count: usize },

    #[snafu(display("Expect a metric matcher, but not found"))]
    NoMetricMatcher,

    #[snafu(display("Invalid function argument for {}", fn_name))]
    FunctionInvalidArgument { fn_name: String },

    #[snafu(display(
        "Attempt to combine two tables with different column sets, left: {:?}, right: {:?}",
        left,
        right
    ))]
    CombineTableColumnMismatch {
        left: Vec<String>,
        right: Vec<String>,
    },
}

impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;
        match self {
            TimeIndexNotFound { .. }
            | ValueNotFound { .. }
            | UnsupportedExpr { .. }
            | ExpectRangeSelector { .. }
            | ZeroRangeSelector { .. }
            | ColumnNotFound { .. }
            | Deserialize { .. }
            | FunctionInvalidArgument { .. }
            | UnsupportedVectorMatch { .. }
            | CombineTableColumnMismatch { .. }
            | DataFusionPlanning { .. }
            | UnexpectedPlanExpr { .. }
            | IllegalRange { .. } => StatusCode::InvalidArguments,

            UnknownTable { .. } | EmptyRange { .. } => StatusCode::Internal,

            TableNameNotFound { .. } => StatusCode::TableNotFound,

            MultipleMetricMatchers { .. } | NoMetricMatcher { .. } => StatusCode::InvalidSyntax,

            Catalog { source, .. } => source.status_code(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for DataFusionError {
    fn from(err: Error) -> Self {
        DataFusionError::External(Box::new(err))
    }
}