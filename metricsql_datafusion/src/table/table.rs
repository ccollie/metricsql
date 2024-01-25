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
use std::sync::Arc;
use async_trait::async_trait;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::logical_expr::Expr;
use datafusion_expr::TableType;

use crate::common::recordbatch::SendableRecordBatchStream;
use crate::table::metadata::{FilterPushDownType, TableInfoRef};
use crate::table::storage::ScanRequest;

use super::error::Result;

pub mod adapter;
mod scan;
pub mod numbers;
mod metrics;

/// Table abstraction.
#[async_trait]
pub trait Table: Send + Sync {
    /// Returns the table as [`Any`](std::any::Any) so that it can be
    /// downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    /// Get a reference to the schema for this table
    fn schema(&self) -> SchemaRef;

    /// Get a reference to the table info.
    fn table_info(&self) -> TableInfoRef;

    /// Get the type of this table for metadata/catalog purposes.
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan_to_stream(&self, request: ScanRequest) -> Result<SendableRecordBatchStream>;

    /// Tests whether the table provider can make use of any or all filter expressions
    /// to optimise data retrieval.
    fn supports_filters_pushdown(&self, filters: &[&Expr]) -> Result<Vec<FilterPushDownType>> {
        Ok(vec![FilterPushDownType::Unsupported; filters.len()])
    }

    /// Close the table.
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

pub type TableRef = Arc<dyn Table>;