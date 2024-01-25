use std::sync::Arc;
use async_trait::async_trait;
use futures_util::future::try_join_all;
use metricsql_engine::{
    Deadline,
    MetricStorage,
    QueryResults,
    RuntimeResult,
    SearchQuery
};
use crate::query::datafusion::DatafusionQueryEngine;
use crate::query::planner::EvalStmt;
use crate::session::context::QueryContext;

pub struct DatafusionProvider {
    engine: Arc<DatafusionQueryEngine>,
    disallow_cross_schema_query: bool,
    default_catalog: String,
    default_schema: String,
}

impl DatafusionProvider {
    pub fn new(engine: Arc<DatafusionQueryEngine>,
               disallow_cross_schema_query: bool,
               default_catalog: String,
               default_schema: String) -> Self {
        Self {
            engine,
            disallow_cross_schema_query,
            default_catalog,
            default_schema,
        }
    }
}

#[async_trait]
impl MetricStorage for DatafusionProvider {
    async fn search(&self, sq: &SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        let query_ctx = QueryContext::with(
            &self.default_catalog,
            &self.default_schema
        );

        let mut calls = Vec::with_capacity(sq.matchers.len());
        for filters in sq.matchers.into_iter() {
            let stmt = EvalStmt {
                start: sq.start,
                end: sq.end,
                filters
            };
            let call = self.engine.exec_stmt(stmt, query_ctx.clone());
            calls.push(call);
        }

        let series = try_join_all(calls)
            .await?
            .into_iter()
            .collect::<Vec<_>>();


    }
}
