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

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::dataframe::DataFrame;
use datafusion_expr::{LogicalPlan, ScalarUDF};

use crate::catalog::CatalogManagerRef;
use crate::query::error::Result;
use crate::query::datafusion::DatafusionQueryEngine;
use crate::query::Output;
use crate::query::planner::LogicalPlanner;
use crate::query::query_engine::state::QueryEngineState;
use crate::session::context::QueryContextRef;
use crate::table::TableRef;

/// Describe statement result
#[derive(Debug)]
pub struct DescribeResult {
    /// The schema of statement
    pub schema: SchemaRef,
    /// The logical plan for statement
    pub logical_plan: LogicalPlan,
}

#[async_trait]
pub trait QueryEngine: Send + Sync {
    /// Returns the query query_engine as Any
    /// so that it can be downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    fn planner(&self) -> Arc<dyn LogicalPlanner>;

    fn name(&self) -> &str;

    async fn describe(&self, plan: LogicalPlan) -> Result<DescribeResult>;

    async fn execute(&self, plan: LogicalPlan, query_ctx: QueryContextRef) -> Result<Output>;

    fn register_udf(&self, udf: ScalarUDF);

    /// Create a DataFrame from a table.
    fn read_table(&self, table: TableRef) -> Result<DataFrame>;
}

pub struct QueryEngineFactory {
    query_engine: Arc<dyn QueryEngine>,
}

impl QueryEngineFactory {

    pub fn new(
        catalog_manager: CatalogManagerRef,
    ) -> Self {
        let state = Arc::new(QueryEngineState::new(
            catalog_manager,
        ));
        let query_engine = Arc::new(DatafusionQueryEngine::new(state));
        register_functions(&query_engine);
        Self { query_engine }
    }

    pub fn query_engine(&self) -> QueryEngineRef {
        self.query_engine.clone()
    }
}

fn register_functions(_query_engine: &Arc<DatafusionQueryEngine>) {
}

pub type QueryEngineRef = Arc<dyn QueryEngine>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_engine_factory() {
        let catalog_list = crate::catalog::memory::new_memory_catalog_manager().unwrap();
        let factory = QueryEngineFactory::new(catalog_list);

        let engine = factory.query_engine();

        assert_eq!("datafusion", engine.name());
    }
}
