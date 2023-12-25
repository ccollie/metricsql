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

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use datafusion::execution::context::SessionState;
use datafusion_expr::LogicalPlan;
use snafu::ResultExt;

use metricsql_common::prelude::BoxedError;
use metricsql_parser::label::Matchers;

use crate::catalog::table_source::DfTableSourceProvider;
use crate::planner::PromPlanner;
use crate::query::error::{QueryPlanSnafu, Result};
use crate::query::query_engine::QueryEngineState;
use crate::session::context::QueryContextRef;

#[derive(Debug, Clone)]
pub struct EvalStmt {
    /// The time boundaries for the evaluation. If start equals end an instant
    /// is evaluated.
    pub start: SystemTime,
    pub end: SystemTime,
    pub filters: Vec<Matchers>,
}

#[async_trait]
pub trait LogicalPlanner: Send + Sync {
    async fn plan(&self, stmt: EvalStmt, query_ctx: QueryContextRef) -> Result<LogicalPlan>;
}

pub struct DfLogicalPlanner {
    engine_state: Arc<QueryEngineState>,
    session_state: SessionState,
}

impl DfLogicalPlanner {
    pub fn new(engine_state: Arc<QueryEngineState>) -> Self {
        let session_state = engine_state.session_state();
        Self {
            engine_state,
            session_state,
        }
    }

    #[tracing::instrument(skip_all)]
    async fn plan_pql(&self, stmt: EvalStmt, query_ctx: QueryContextRef) -> Result<LogicalPlan> {
        let table_provider = DfTableSourceProvider::new(
            self.engine_state.catalog_manager().clone(),
            self.engine_state.disallow_cross_schema_query(),
            query_ctx.as_ref(),
        );
        PromPlanner::stmt_to_plan(table_provider, stmt)
            .await
            .map_err(BoxedError::new)
            .context(QueryPlanSnafu)
    }
}

#[async_trait]
impl LogicalPlanner for DfLogicalPlanner {
    #[tracing::instrument(skip_all)]
    async fn plan(&self, stmt: EvalStmt, query_ctx: QueryContextRef) -> Result<LogicalPlan> {
        self.plan_pql(stmt, query_ctx).await
    }
}
