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

//! Planner, QueryEngine implementations based on DataFusion.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::physical_plan::analyze::AnalyzeExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion_common::ResolvedTableReference;
use datafusion_expr::{Expr, LogicalPlan, ScalarUDF};
use snafu::{OptionExt, ResultExt};

use metricsql_common::error::ext::BoxedError;

use crate::common::recordbatch::adapter::RecordBatchStreamAdapter;
use crate::common::recordbatch::{
    EmptyRecordBatchStream, RecordBatch, RecordBatches, SendableRecordBatchStream,
};
use crate::query::datafusion::error::{DatafusionSnafu, PhysicalPlanDowncastSnafu};
use crate::query::error::{
    CatalogSnafu, CreateRecordBatchSnafu, DataFusionSnafu, QueryExecutionSnafu, Result,
    TableNotFoundSnafu,
};
use crate::query::executor::QueryExecutor;
use crate::query::logical_optimizer::LogicalOptimizer;
use crate::query::metrics::{
    METRIC_CREATE_PHYSICAL_ELAPSED, METRIC_EXEC_PLAN_ELAPSED, METRIC_OPTIMIZE_LOGICAL_ELAPSED,
    METRIC_OPTIMIZE_PHYSICAL_ELAPSED,
};
use crate::query::physical_optimizer::PhysicalOptimizer;
use crate::query::physical_plan::{DfPhysicalPlanAdapter, PhysicalPlan, PhysicalPlanAdapter};
use crate::query::physical_planner::PhysicalPlanner;
use crate::query::planner::{DfLogicalPlanner, EvalStmt, LogicalPlanner};
use crate::query::query_engine::{
    DescribeResult, QueryEngine, QueryEngineContext, QueryEngineState,
};
use crate::query::Output;
use crate::session::context::{QueryContextRef};
use crate::table::TableRef;

mod error;
mod planner;

pub struct DatafusionQueryEngine {
    state: Arc<QueryEngineState>,
}

impl DatafusionQueryEngine {
    pub fn new(state: Arc<QueryEngineState>) -> Self {
        Self { state }
    }

    pub async fn exec_stmt(&self, stmt: EvalStmt, query_ctx: QueryContextRef) -> Result<Output> {
        let plan = self
            .planner()
            .plan(stmt, query_ctx.clone())
            .await?;

        self.execute(plan, query_ctx).await
    }

    #[tracing::instrument(skip_all)]
    async fn exec_query_plan(
        &self,
        plan: LogicalPlan,
        query_ctx: QueryContextRef,
    ) -> Result<Output> {
        let mut ctx = QueryEngineContext::new(self.state.session_state(), query_ctx.clone());

        // `create_physical_plan` will optimize logical plan internally
        let physical_plan = self.create_physical_plan(&mut ctx, &plan).await?;
        let physical_plan = self.optimize_physical_plan(&mut ctx, physical_plan)?;

        Ok(Output::Stream(self.execute_stream(&ctx, &physical_plan)?))
    }

    async fn find_table(&self, table_name: &ResolvedTableReference<'_>) -> Result<TableRef> {
        let catalog_name = table_name.catalog.as_ref();
        let schema_name = table_name.schema.as_ref();
        let table_name = table_name.table.as_ref();

        self.state
            .catalog_manager()
            .table(catalog_name, schema_name, table_name)
            .await
            .context(CatalogSnafu)?
            .with_context(|| TableNotFoundSnafu { table: table_name })
    }

}

#[async_trait]
impl QueryEngine for DatafusionQueryEngine {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn planner(&self) -> Arc<dyn LogicalPlanner> {
        Arc::new(DfLogicalPlanner::new(self.state.clone()))
    }

    fn name(&self) -> &str {
        "datafusion"
    }

    async fn describe(&self, plan: LogicalPlan) -> Result<DescribeResult> {
        let optimised_plan = self.optimize(&plan)?;
        let schena = optimised_plan.schema();
        Ok(DescribeResult {
            schema: schena.clone(),
            logical_plan: optimised_plan,
        })
    }

    async fn execute(&self, plan: LogicalPlan, query_ctx: QueryContextRef) -> Result<Output> {
        if matches!(plan, LogicalPlan::Dml(_)) {
            return self.exec_dml_statement(plan, query_ctx).await;
        }
        self.exec_query_plan(plan, query_ctx).await
    }

    fn register_udf(&self, udf: ScalarUDF) {
        self.state.register_udf(udf);
    }
    fn read_table(&self, table: TableRef) -> Result<DataFrame> {
        Ok(DataFrame::DataFusion(
            self.state
                .read_table(table)
                .context(DatafusionSnafu)
                .map_err(BoxedError::new)
                .context(QueryExecutionSnafu)?,
        ))
    }
}

impl LogicalOptimizer for DatafusionQueryEngine {
    #[tracing::instrument(skip_all)]
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        let _timer = METRIC_OPTIMIZE_LOGICAL_ELAPSED.start_timer();
        self.state
            .session_state()
            .optimize(plan)
            .context(DatafusionSnafu)
            .map_err(BoxedError::new)
            .context(QueryExecutionSnafu)
    }
}

impl PhysicalPlanner for DatafusionQueryEngine {
    #[tracing::instrument(skip_all)]
    async fn create_physical_plan(
        &self,
        ctx: &mut QueryEngineContext,
        logical_plan: &LogicalPlan,
    ) -> Result<Arc<dyn PhysicalPlan>> {
        let _timer = METRIC_CREATE_PHYSICAL_ELAPSED.start_timer();
        let state = ctx.state();
        let physical_plan = state
            .create_physical_plan(logical_plan)
            .await
            .context(DatafusionSnafu)
            .map_err(BoxedError::new)
            .context(QueryExecutionSnafu)?;

        Ok(Arc::new(PhysicalPlanAdapter::new(
            physical_plan
                .schema()
                .clone(),
            physical_plan,
        )))
    }
}

impl PhysicalOptimizer for DatafusionQueryEngine {
    #[tracing::instrument(skip_all)]
    fn optimize_physical_plan(
        &self,
        ctx: &mut QueryEngineContext,
        plan: Arc<dyn PhysicalPlan>,
    ) -> Result<Arc<dyn PhysicalPlan>> {
        let _timer = METRIC_OPTIMIZE_PHYSICAL_ELAPSED.start_timer();

        let state = ctx.state();
        let config = state.config_options();
        let df_plan = plan
            .as_any()
            .downcast_ref::<PhysicalPlanAdapter>()
            .context(PhysicalPlanDowncastSnafu)
            .map_err(BoxedError::new)
            .context(QueryExecutionSnafu)?
            .df_plan();

        // skip optimize AnalyzeExec plan
        let optimized_plan =
            if let Some(analyze_plan) = df_plan.as_any().downcast_ref::<AnalyzeExec>() {
                let mut new_plan = analyze_plan.input().clone();
                for optimizer in state.physical_optimizers() {
                    new_plan = optimizer
                        .optimize(new_plan, config)
                        .context(DataFusionSnafu)?;
                }
                Arc::new(analyze_plan.clone())
                    .with_new_children(vec![new_plan])
                    .unwrap()
            } else {
                let mut new_plan = df_plan;
                for optimizer in state.physical_optimizers() {
                    new_plan = optimizer
                        .optimize(new_plan, config)
                        .context(DataFusionSnafu)?;
                }
                new_plan
            };

        Ok(Arc::new(PhysicalPlanAdapter::new(
            plan.schema(),
            optimized_plan,
        )))
    }
}

impl QueryExecutor for DatafusionQueryEngine {
    #[tracing::instrument(skip_all)]
    fn execute_stream(
        &self,
        ctx: &QueryEngineContext,
        plan: &Arc<dyn PhysicalPlan>,
    ) -> Result<SendableRecordBatchStream> {
        let _timer = METRIC_EXEC_PLAN_ELAPSED.start_timer();
        let task_ctx = ctx.build_task_ctx();

        match plan.output_partitioning().partition_count() {
            0 => Ok(Box::pin(EmptyRecordBatchStream::new(plan.schema()))),
            1 => Ok(plan
                .execute(0, task_ctx)
                .context(error::ExecutePhysicalPlanSnafu)
                .map_err(BoxedError::new)
                .context(QueryExecutionSnafu))?,
            _ => {
                // merge into a single partition
                let plan =
                    CoalescePartitionsExec::new(Arc::new(DfPhysicalPlanAdapter(plan.clone())));
                // CoalescePartitionsExec must produce a single partition
                assert_eq!(1, plan.output_partitioning().partition_count());
                let df_stream = plan
                    .execute(0, task_ctx)
                    .context(DatafusionSnafu)
                    .map_err(BoxedError::new)
                    .context(QueryExecutionSnafu)?;
                let stream = RecordBatchStreamAdapter::try_new(df_stream)
                    .context(error::ConvertDfRecordBatchStreamSnafu)
                    .map_err(BoxedError::new)
                    .context(QueryExecutionSnafu)?;
                Ok(Box::pin(stream))
            }
        }
    }
}

/// Creates a table in memory and executes a show statement on the table.
pub async fn execute_show_with_filter(
    record_batch: RecordBatch,
    filter: Option<Expr>,
) -> Result<Output> {
    let table_name = "table_name";
    let context = SessionContext::new();
    let schema = record_batch.schema.clone();

    context
        .register_batch(table_name, record_batch.into_df_record_batch())
        .context(DatafusionSnafu)
        .map_err(BoxedError::new)
        .context(QueryExecutionSnafu)?;
    let mut dataframe = context
        .sql(&format!("SELECT * FROM {table_name}"))
        .await
        .context(DatafusionSnafu)
        .map_err(BoxedError::new)
        .context(QueryExecutionSnafu)?;
    if let Some(filter) = filter {
        dataframe = dataframe
            .filter(filter)
            .context(DatafusionSnafu)
            .map_err(BoxedError::new)
            .context(QueryExecutionSnafu)?
    }
    let df_batches = dataframe
        .collect()
        .await
        .context(DatafusionSnafu)
        .map_err(BoxedError::new)
        .context(QueryExecutionSnafu)?;

    let mut batches = Vec::with_capacity(df_batches.len());
    for df_batch in df_batches.into_iter() {
        let batch = RecordBatch::try_from_df_record_batch(schema.clone(), df_batch)
            .context(CreateRecordBatchSnafu)?;
        batches.push(batch);
    }
    let record_batches = RecordBatches::try_new(schema.clone(), batches).context(CreateRecordBatchSnafu)?;
    Ok(Output::RecordBatches(record_batches))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow::Borrowed;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Schema};
    use datafusion::prelude::{col, lit};
    use datafusion::sql::parser::DFParser as QueryLanguageParser;
    use datafusion::sql::sqlparser::ast::Statement::ShowTables;
    use datafusion_expr::LogicalPlan::Statement;

    use crate::catalog::consts::{
        DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME, NUMBERS_TABLE_ID, NUMBERS_TABLE_NAME,
    };
    use crate::catalog::{RegisterTableRequest, StringVectorBuilder};
    use crate::common::recordbatch::{util, RecordBatch};
    use crate::query::query_engine::{QueryEngineFactory, QueryEngineRef};
    use crate::query::Output;
    use crate::session::context::QueryContext;
    use crate::sql::statements::show::ShowKind;
    use crate::table::numbers::NumbersTable;
    use crate::table::schema::column_schema::Field;
    use crate::table::table::numbers::{NumbersTable, NUMBERS_TABLE_NAME};

    use super::*;

    async fn create_test_engine() -> QueryEngineRef {
        let catalog_manager = crate::catalog::memory::new_memory_catalog_manager().unwrap();
        let req = RegisterTableRequest {
            catalog: DEFAULT_CATALOG_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table_name: NUMBERS_TABLE_NAME.to_string(),
            table: NumbersTable::table(NUMBERS_TABLE_ID),
        };
        catalog_manager.register_table_sync(req).unwrap();

        QueryEngineFactory::new(catalog_manager).query_engine()
    }

    #[tokio::test]
    async fn test_execute() {
        let engine = create_test_engine().await;
        let sql = "select sum(number) from numbers limit 20";

        let stmt = QueryLanguageParser::parse_sql(sql).unwrap();
        let plan = engine
            .planner()
            .plan(stmt, QueryContext::arc())
            .await
            .unwrap();

        let output = engine.execute(plan, QueryContext::arc()).await.unwrap();

        match output {
            Output::Stream(record_batch) => {
                let numbers = util::collect(record_batch).await.unwrap();
                assert_eq!(1, numbers.len());
                assert_eq!(numbers[0].num_columns(), 1);
                assert_eq!(1, numbers[0].schema.num_columns());
                assert_eq!(
                    "SUM(numbers.number)",
                    numbers[0].schema.column_schemas()[0].name
                );

                let batch = &numbers[0];
                assert_eq!(1, batch.num_columns());
                assert_eq!(batch.column(0).len(), 1);

                assert_eq!(
                    *batch.column(0),
                    Arc::new(UInt64Array::from_slice([4950])) as ArrayRef
                );
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn test_read_table() {
        let engine = create_test_engine().await;

        let engine = engine
            .as_any()
            .downcast_ref::<DatafusionQueryEngine>()
            .unwrap();
        let table = engine
            .find_table(&ResolvedTableReference {
                catalog: Borrowed("greptime"),
                schema: Borrowed("public"),
                table: Borrowed("numbers"),
            })
            .await
            .unwrap();

        let DataFrame::DataFusion(df) = engine.read_table(table).unwrap();
        let df = df
            .select_columns(&["number"])
            .unwrap()
            .filter(col("number").lt(lit(10)))
            .unwrap();
        let batches = df.collect().await.unwrap();
        assert_eq!(1, batches.len());
        let batch = &batches[0];

        assert_eq!(1, batch.num_columns());
        assert_eq!(batch.column(0).len(), 10);

        assert_eq!(
            batch.column(0).unwrap(),
            Arc::new(UInt32Array::from_slice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) as ArrayRef
        );
    }

    #[tokio::test]
    async fn test_describe() {
        let engine = create_test_engine().await;
        let sql = "select sum(number) from numbers limit 20";

        let stmt = QueryLanguageParser::parse_sql(sql).unwrap();

        let plan = engine
            .planner()
            .plan(stmt, QueryContext::arc())
            .await
            .unwrap();

        let DescribeResult {
            schema,
            logical_plan,
        } = engine.describe(plan).await.unwrap();

        assert_eq!(
            schema.column_schemas()[0],
            Field::new("SUM(numbers.number)", DataType::UInt64, true)
        );
        assert_eq!("Limit: skip=0, fetch=20\n  Aggregate: groupBy=[[]], aggr=[[SUM(CAST(numbers.number AS UInt64))]]\n    TableScan: numbers projection=[number]", format!("{}", logical_plan.display_indent()));
    }

    #[tokio::test]
    async fn test_show_tables() {
        // No filter
        let column_schemas = vec![Field::new("Tables", DataType::Utf8, false)];
        let schema = Arc::new(Schema::new(column_schemas));
        let mut builder = StringVectorBuilder::with_capacity(3);
        builder.push(Some("monitor"));
        builder.push(Some("system_metrics"));
        let columns = vec![builder.to_vector()];
        let record_batch = RecordBatch::new(schema, columns).unwrap();
        let output = execute_show_with_filter(record_batch, None).await.unwrap();
        let Output::RecordBatches(record_batches) = output else {
            unreachable!()
        };
        let expected = "\
+----------------+
| Tables         |
+----------------+
| monitor        |
| system_metrics |
+----------------+";
        assert_eq!(record_batches.pretty_print().unwrap(), expected);

        // Filter
        let column_schemas = vec![Field::new("Tables", DataType::Utf8, false)];
        let schema = Arc::new(Schema::new(column_schemas));
        let mut builder = StringVectorBuilder::with_capacity(3);
        builder.push(Some("monitor"));
        builder.push(Some("system_metrics"));
        let columns = vec![builder.to_vector()];
        let record_batch = RecordBatch::new(schema, columns).unwrap();
        let statement = ParserContext::create_with_dialect(
            "SHOW TABLES WHERE \"Tables\"='monitor'",
            &GreptimeDbDialect {},
        )
        .unwrap()[0]
            .clone();
        let Statement::ShowTables(ShowTables { kind, .. }) = statement else {
            unreachable!()
        };
        let ShowKind::Where(filter) = kind else {
            unreachable!()
        };
        let output = execute_show_with_filter(record_batch, Some(filter))
            .await
            .unwrap();
        let Output::RecordBatches(record_batches) = output else {
            unreachable!()
        };
        let expected = "\
+---------+
| Tables  |
+---------+
| monitor |
+---------+";
        assert_eq!(record_batches.pretty_print().unwrap(), expected);
    }
}
