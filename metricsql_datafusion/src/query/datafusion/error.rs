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

use crate::datatypes::error::DataTypeError;

/// Inner error of datafusion based query query_engine.
#[derive(Snafu)]
#[snafu(visibility(pub))]
#[derive(Debug)]
pub enum InnerError {
    #[snafu(display("DataFusion error"))]
    Datafusion {
        #[snafu(source)]
        error: DataFusionError,
    },

    #[snafu(display("PhysicalPlan downcast failed"))]
    PhysicalPlanDowncast,

    #[snafu(display("Fail to convert arrow schema"))]
    ConvertSchema {
        source: DataTypeError,
    },

    #[snafu(display("Failed to convert DataFusion's recordbatch stream"))]
    ConvertDfRecordBatchStream {
        source: crate::common::recordbatch::error::Error,
    },

    #[snafu(display("Failed to execute physical plan"))]
    ExecutePhysicalPlan {
        source: crate::query::error::Error,
    },
}

impl ErrorExt for InnerError {
    fn status_code(&self) -> StatusCode {
        use InnerError::*;

        match self {
            // TODO(yingwen): Further categorize datafusion error.
            Datafusion { .. } => StatusCode::EngineExecuteQuery,
            PhysicalPlanDowncast { .. } | ConvertSchema { .. } => StatusCode::Unexpected,
            ConvertDfRecordBatchStream { source, .. } => source.status_code(),
            ExecutePhysicalPlan { source, .. } => source.status_code(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
