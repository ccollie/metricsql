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

use std::collections::HashSet;
use datafusion::common::{OwnedTableReference, Result as DfResult};
use datafusion::datasource::DefaultTableSource;
use datafusion::logical_expr::{BinaryExpr, LogicalPlan, LogicalPlanBuilder, Operator};
use datafusion_expr::expr::Alias;
use datafusion::prelude::{Column, Expr as DfExpr};
use datafusion::scalar::ScalarValue;
use datafusion::sql::TableReference;
use datafusion_expr::utils::conjunction;
use snafu::{ensure, OptionExt, ResultExt};
use metricsql_parser::prelude::{LabelFilter, LabelFilterOp, Matchers, METRIC_NAME};

use crate::catalog::table_source::DfTableSourceProvider;
use crate::query::planner::EvalStmt;
use crate::table::adapter::DfTableProviderAdapter;
use crate::table::is_timestamp_field;

use super::error::{
    CatalogSnafu, ColumnNotFoundSnafu, DataFusionPlanningSnafu, MultipleMetricMatchersSnafu,
    Result, TableNameNotFoundSnafu, TimeIndexNotFoundSnafu,
    UnknownTableSnafu, UnsupportedExprSnafu, ValueNotFoundSnafu,
};

/// Special modifier to project field columns under multi-field mode
const FIELD_COLUMN_MATCHER: &str = "__field__";

#[derive(Default, Debug, Clone)]
struct PromPlannerContext {
    // query parameters
    start: i64,
    end: i64,

    // planner states
    table_name: Option<String>,
    time_index_column: Option<String>,
    field_columns: Vec<String>,
    tag_columns: Vec<String>,
    field_column_matcher: Option<Vec<LabelFilter>>,
}

impl PromPlannerContext {
    fn from_eval_stmt(eval_stmt: &EvalStmt) -> Self {
        Self {
            start: eval_stmt.start,
            end: eval_stmt.end,
            ..Default::default()
        }
    }

    /// Reset all planner states
    fn reset(&mut self) {
        self.table_name = None;
        self.time_index_column = None;
        self.field_columns = vec![];
        self.tag_columns = vec![];
        self.field_column_matcher = None;
    }
}

pub(crate) struct PromPlanner {
    table_provider: DfTableSourceProvider,
    ctx: PromPlannerContext,
}

impl PromPlanner {
    pub async fn query_to_plan(
        table_provider: DfTableSourceProvider,
        stmt: &EvalStmt,
    ) -> Result<LogicalPlan> {
        let mut planner = Self {
            table_provider,
            ctx: PromPlannerContext::from_eval_stmt(&stmt),
        };
        planner.eval_stmt_to_plan(stmt).await
    }

    pub(crate) fn new(start: i64, end: i64, table_provider: DfTableSourceProvider) -> Self {
        Self {
            table_provider,
            ctx: PromPlannerContext {
                start,
                end,
                ..Default::default()
            },
        }
    }

    pub async fn eval_stmt_to_plan(&mut self, stmt: &EvalStmt) -> Result<LogicalPlan> {
        self.ctx.start = stmt.start;
        self.ctx.end = stmt.end;

        let name: Option<String> = None;
        let matchers = self.preprocess_label_matchers(&stmt.filters, &name)?;
        self.setup_context().await?;

        let normalize = self
            .selector_to_series_normalize_plan(matchers)
            .await?;

        Ok(normalize)
    }

    /// Extract metric name from `__name__` matcher and set it into [PromPlannerContext].
    /// Returns a new [Matchers] that doesn't contains metric name matcher.
    ///
    /// Each call to this function means new selector is started. Thus the context will be reset
    /// at first.
    ///
    /// Name rule:
    /// - if `name` is some, then the matchers MUST NOT contains `__name__` matcher.
    /// - if `name` is none, then the matchers MAY contains NONE OR MULTIPLE `__name__` matchers.
    fn preprocess_label_matchers(
        &mut self,
        label_matchers: &Matchers,
        name: &Option<String>,
    ) -> Result<Matchers> {
        self.ctx.reset();
        let metric_name;

        let matchers = label_matchers.find_matchers(METRIC_NAME);
        if let Some(name) = name.clone() {
            metric_name = Some(name);
            ensure!(
                matchers.len() == 1,
                MultipleMetricMatchersSnafu { count: matchers.len( )}
            );
        } else {
            let matchers = label_matchers.find_matchers(METRIC_NAME);
            ensure!(
                matchers.len() == 1,
                MultipleMetricMatchersSnafu { count: matchers.len() }
            );
            metric_name = Some(matchers[0].value.clone());
        }
        self.ctx.table_name = metric_name;

        let mut matchers = HashSet::new();
        for matcher in label_matchers.iter() {
            if matcher.label == FIELD_COLUMN_MATCHER {
                self.ctx
                    .field_column_matcher
                    .get_or_insert_with(|| Default::default())
                    .push(matcher.clone());
            } else {
                let _ = matchers.insert(matcher.clone());
            }
        }
        let matchers = matchers.into_iter().collect();
        Ok(Matchers::new(matchers))
    }

    async fn selector_to_series_normalize_plan(
        &mut self,
        label_matchers: Matchers,
    ) -> Result<LogicalPlan> {
        let table_name = self.ctx.table_name.clone().unwrap();

        let mut scan_filters = self.matchers_to_expr(label_matchers.clone());
        scan_filters.push(self.create_time_index_column_expr()?.gt_eq(DfExpr::Literal(
            ScalarValue::TimestampMillisecond(Some(self.ctx.start), None),
        )));
        scan_filters.push(self.create_time_index_column_expr()?.lt_eq(DfExpr::Literal(
            ScalarValue::TimestampMillisecond(Some(self.ctx.end), None),
        )));

        // make table scan with filter expressions
        let mut table_scan = self
            .create_table_scan_plan(&table_name, scan_filters.clone())
            .await?;

        // make a projection plan if there is any `__field__` matcher
        if let Some(field_matchers) = &self.ctx.field_column_matcher {
            let col_set = self.ctx.field_columns.iter().collect::<HashSet<_>>();
            // opt-in set
            let mut result_set = HashSet::new();
            // opt-out set
            let mut reverse_set = HashSet::new();
            for matcher in field_matchers {
                match &matcher.op {
                    LabelFilterOp::Equal => {
                        if col_set.contains(&matcher.value) {
                            let _ = result_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                            }
                            .build());
                        }
                    }
                    LabelFilterOp::NotEqual => {
                        if col_set.contains(&matcher.value) {
                            let _ = reverse_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                            }
                            .build());
                        }
                    }
                    LabelFilterOp::RegexEqual => {
                        let regex = regex::Regex::new(matcher.value.as_str()).unwrap();
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
                                let _ = result_set.insert(col.clone());
                            }
                        }
                    }
                    LabelFilterOp::RegexNotEqual => {
                        let re = regex::Regex::new(matcher.value.as_str()).unwrap();
                        for col in &self.ctx.field_columns {
                            if re.is_match(col) {
                                let _ = reverse_set.insert(col.clone());
                            }
                        }
                    }
                }
            }
            // merge two set
            if result_set.is_empty() {
                result_set = col_set.into_iter().cloned().collect();
            }
            for col in reverse_set {
                let _ = result_set.remove(&col);
            }

            // mask the field columns in context using computed result set
            self.ctx.field_columns = self
                .ctx
                .field_columns
                .drain(..)
                .filter(|col| result_set.contains(col))
                .collect();

            let exprs = result_set
                .into_iter()
                .map(|col| DfExpr::Column(col.into()))
                .chain(self.create_tag_column_exprs()?)
                .chain(Some(self.create_time_index_column_expr()?))
                .collect::<Vec<_>>();
            // reuse this variable for simplicity
            table_scan = LogicalPlanBuilder::from(table_scan)
                .project(exprs)
                .context(DataFusionPlanningSnafu)?
                .build()
                .context(DataFusionPlanningSnafu)?;
        }

        // make filter and sort plan
        let mut plan_builder = LogicalPlanBuilder::from(table_scan);
        let accurate_filters = self.matchers_to_expr(label_matchers);
        if !accurate_filters.is_empty() {
            plan_builder = ResultExt::context(
                plan_builder.filter(conjunction(accurate_filters).unwrap()),
                DataFusionPlanningSnafu
            )?;
        }
        let logical_plan = plan_builder
            .sort(self.create_tag_and_time_index_column_sort_exprs()?)
            .context(DataFusionPlanningSnafu)?
            .build()
            .context(DataFusionPlanningSnafu)?;

        Ok(logical_plan)
    }

    // TODO(ruihang): ignore `MetricNameLabel` (`__name__`) matcher
    fn matchers_to_expr(&self, label_matchers: Matchers) -> Vec<DfExpr> {
        let mut exprs = Vec::with_capacity(label_matchers.len());
        for matcher in label_matchers.iter() {
            let col = DfExpr::Column(Column::from_name(matcher.label.clone()));
            let lit = DfExpr::Literal(ScalarValue::Utf8(Some(matcher.value.clone())));
            let expr = match matcher.op {
                LabelFilterOp::Equal => col.eq(lit),
                LabelFilterOp::NotEqual => col.not_eq(lit),
                LabelFilterOp::RegexEqual => DfExpr::BinaryExpr(BinaryExpr {
                    left: Box::new(col),
                    op: Operator::RegexMatch,
                    right: Box::new(lit),
                }),
                LabelFilterOp::RegexNotEqual => DfExpr::BinaryExpr(BinaryExpr {
                    left: Box::new(col),
                    op: Operator::RegexNotMatch,
                    right: Box::new(lit),
                }),
            };
            exprs.push(expr);
        }

        exprs
    }

    async fn create_table_scan_plan(
        &mut self,
        table_name: &str,
        filter: Vec<DfExpr>,
    ) -> Result<LogicalPlan> {
        let table_ref = OwnedTableReference::bare(table_name.to_string());
        let provider = self
            .table_provider
            .resolve_table(table_ref.clone())
            .await
            .context(CatalogSnafu)?;

        let result = LogicalPlanBuilder::scan_with_filters(table_ref, provider, None, filter)
            .context(DataFusionPlanningSnafu)?
            .build()
            .context(DataFusionPlanningSnafu)?;

        Ok(result)
    }

    /// Setup [PromPlannerContext]'s state fields.
    async fn setup_context(&mut self) -> Result<()> {
        let table_name = self
            .ctx
            .table_name
            .clone()
            .context(TableNameNotFoundSnafu)?;

        let table = self
            .table_provider
            .resolve_table(TableReference::bare(&table_name))
            .await
            .context(CatalogSnafu)?
            .as_any()
            .downcast_ref::<DefaultTableSource>()
            .context(UnknownTableSnafu)?
            .table_provider
            .as_any()
            .downcast_ref::<DfTableProviderAdapter>()
            .context(UnknownTableSnafu)? // todo: add name to error
            .table();

        // set time index column name
        let time_index = table
            .schema()
            .fields
            .iter()
            .find(|field| is_timestamp_field(*field))
            .with_context(|| TimeIndexNotFoundSnafu { table: table_name.clone() })?
            .name()
            .clone();

        self.ctx.time_index_column = Some(time_index);

        // set values columns
        let values = table
            .table_info()
            .meta
            .field_column_names()
            .cloned()
            .collect();

        self.ctx.field_columns = values;

        // set primary key (tag) columns
        let tags = table
            .table_info()
            .meta
            .row_key_column_names()
            .cloned()
            .collect();

        self.ctx.tag_columns = tags;

        Ok(())
    }

    fn create_time_index_column_expr(&self) -> Result<DfExpr> {
        Ok(DfExpr::Column(Column::from_name(
            self.ctx
                .time_index_column
                .clone()
                .with_context(|| TimeIndexNotFoundSnafu { table: "unknown" })?,
        )))
    }

    fn create_tag_column_exprs(&self) -> Result<Vec<DfExpr>> {
        let mut result = Vec::with_capacity(self.ctx.tag_columns.len());
        for tag in &self.ctx.tag_columns {
            let expr = DfExpr::Column(Column::from_name(tag));
            result.push(expr);
        }
        Ok(result)
    }

    fn create_tag_and_time_index_column_sort_exprs(&self) -> Result<Vec<DfExpr>> {
        let mut result = self
            .ctx
            .tag_columns
            .iter()
            .map(|col| DfExpr::Column(Column::from_name(col)).sort(false, false))
            .collect::<Vec<_>>();
        result.push(self.create_time_index_column_expr()?.sort(false, false));
        Ok(result)
    }

    fn create_empty_values_filter_expr(&self) -> Result<DfExpr> {
        let mut exprs = Vec::with_capacity(self.ctx.field_columns.len());
        for value in &self.ctx.field_columns {
            let expr = DfExpr::Column(Column::from_name(value)).is_not_null();
            exprs.push(expr);
        }

        conjunction(exprs).context(ValueNotFoundSnafu {
            table: self.ctx.table_name.clone().unwrap(),
        })
    }

    /// Build a projection that project and perform operation expr for every value column.
    /// Non-value columns (tag and timestamp) will be preserved in the projection.
    ///
    /// # Side effect
    ///
    /// This function will update the value columns in the context. Those new column names
    /// don't contain qualifier.
    fn projection_for_each_field_column<F>(
        &mut self,
        input: LogicalPlan,
        name_to_expr: F,
    ) -> Result<LogicalPlan>
    where
        F: FnMut(&String) -> Result<DfExpr>,
    {
        let non_field_columns_iter = self
            .ctx
            .tag_columns
            .iter()
            .chain(self.ctx.time_index_column.iter())
            .map(|col| {
                Ok(DfExpr::Column(Column::new(
                    self.ctx.table_name.clone(),
                    col,
                )))
            });

        // build computation expressions
        let result_field_columns = self
            .ctx
            .field_columns
            .iter()
            .map(name_to_expr)
            .collect::<Result<Vec<_>>>()?;

        // alias the computation expressions to remove qualifier
        self.ctx.field_columns = result_field_columns
            .iter()
            .map(|expr| expr.display_name())
            .collect::<DfResult<Vec<_>>>()
            .context(DataFusionPlanningSnafu)?;

        let field_columns_iter = result_field_columns
            .into_iter()
            .zip(self.ctx.field_columns.iter())
            .map(|(expr, name)| Ok(DfExpr::Alias(Alias::new(expr, None::<OwnedTableReference>, name))));

        // chain non-value columns (unchanged) and value columns (applied computation then alias)
        let project_fields = non_field_columns_iter
            .chain(field_columns_iter)
            .collect::<Result<Vec<_>>>()?;

        LogicalPlanBuilder::from(input)
            .project(project_fields)
            .context(DataFusionPlanningSnafu)?
            .build()
            .context(DataFusionPlanningSnafu)
    }

    /// Build a filter plan that filter on value column. Notice that only one value column
    /// is expected.
    fn filter_on_field_column<F>(
        &self,
        input: LogicalPlan,
        mut name_to_expr: F,
    ) -> Result<LogicalPlan>
    where
        F: FnMut(&String) -> Result<DfExpr>,
    {
        ensure!(
            self.ctx.field_columns.len() == 1,
            UnsupportedExprSnafu {
                name: "filter on multi-value input"
            }
        );

        let field_column_filter = name_to_expr(&self.ctx.field_columns[0])?;

        LogicalPlanBuilder::from(input)
            .filter(field_column_filter)
            .context(DataFusionPlanningSnafu)?
            .build()
            .context(DataFusionPlanningSnafu)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    use arrow_schema::{DataType, Schema, TimeUnit};
    use datafusion::datasource::empty::EmptyTable;

    use metricsql_engine::{parse_metric_selector, SearchQuery};
    use metricsql_parser::parser;

    use crate::catalog::consts::{DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME};
    use crate::catalog::memory::MemoryCatalogManager;
    use crate::catalog::RegisterTableRequest;
    use crate::catalog::table_source::DfTableSourceProvider;
    use crate::session::context::QueryContext;
    use crate::table::{TableInfoBuilder, TableMetaBuilder};
    use crate::table::schema::column_schema::Field;

    use super::*;

    async fn build_test_table_provider(
        table_name: String,
        num_tag: usize,
        num_field: usize,
    ) -> DfTableSourceProvider {
        let mut columns = vec![];
        for i in 0..num_tag {
            columns.push(Field::new(format!("tag_{i}"), DataType::Utf8, false));
        }
        columns.push(
            Field::new(
                "timestamp".to_string(),
                DataType::Time64(TimeUnit::Millisecond),
                false,
            )
            .with_time_index(true),
        );
        for i in 0..num_field {
            columns.push(Field::new(
                format!("field_{i}"),
                DataType::Float64,
                true,
            ));
        }
        let schema = Arc::new(Schema::new(columns));
        let table_meta = TableMetaBuilder::default()
            .schema(schema)
            .primary_key_indices((0..num_tag).collect())
            .value_indices((num_tag + 1..num_tag + 1 + num_field).collect())
            .next_column_id(1024)
            .build()
            .unwrap();
        let table_info = TableInfoBuilder::default()
            .name(&table_name)
            .meta(table_meta)
            .build()
            .unwrap();
        let table = EmptyTable::from_table_info(&table_info);
        let catalog_list = MemoryCatalogManager::with_default_setup();
        assert!(catalog_list
            .register_table_sync(RegisterTableRequest {
                catalog: DEFAULT_CATALOG_NAME.to_string(),
                schema: DEFAULT_SCHEMA_NAME.to_string(),
                table_name,
                table,
            })
            .is_ok());

        DfTableSourceProvider::new(catalog_list, false, QueryContext::arc().as_ref())
    }

    async fn indie_query_plan_compare(query: &str, expected: String) {
        let filters = parse_metric_selector(query).unwrap();
        let matchers = Matchers::new(filters);
        let eval_stmt = SearchQuery::new(
            UNIX_EPOCH.into(),
            UNIX_EPOCH.into()
                .checked_add(Duration::from_secs(100_000))
                .unwrap(),
            vec![matchers],
            None,
        );

        let table_provider = build_test_table_provider("some_metric".to_string(), 1, 1).await;
        let plan = PromPlanner::stmt_to_plan(table_provider, eval_stmt)
            .await
            .unwrap();

        assert_eq!(plan.display_indent_schema().to_string(), expected);
    }

    #[tokio::test]
    async fn value_matcher() {
        // template
        let mut eval_stmt = SearchQuery {
            start: UNIX_EPOCH.into(),
            end: UNIX_EPOCH
                .into()
                .checked_add(Duration::from_secs(100_000))
                .unwrap(),
            matchers: vec![],
            max_metrics: 0,
        };

        let cases = [
            // single equal matcher
            (
                r#"some_metric{__field__="field_1"}"#,
                vec![
                    "some_metric.field_1",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // two equal matchers
            (
                r#"some_metric{__field__="field_1", __field__="field_0"}"#,
                vec![
                    "some_metric.field_0",
                    "some_metric.field_1",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // single not_eq matcher
            (
                r#"some_metric{__field__!="field_1"}"#,
                vec![
                    "some_metric.field_0",
                    "some_metric.field_2",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // two not_eq matchers
            (
                r#"some_metric{__field__!="field_1", __field__!="field_2"}"#,
                vec![
                    "some_metric.field_0",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // equal and not_eq matchers (no conflict)
            (
                r#"some_metric{__field__="field_1", __field__!="field_0"}"#,
                vec![
                    "some_metric.field_1",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // equal and not_eq matchers (conflict)
            (
                r#"some_metric{__field__="field_2", __field__!="field_2"}"#,
                vec![
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // single regex eq matcher
            (
                r#"some_metric{__field__=~"field_1|field_2"}"#,
                vec![
                    "some_metric.field_1",
                    "some_metric.field_2",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
            // single regex not_eq matcher
            (
                r#"some_metric{__field__!~"field_1|field_2"}"#,
                vec![
                    "some_metric.field_0",
                    "some_metric.tag_0",
                    "some_metric.tag_1",
                    "some_metric.tag_2",
                    "some_metric.timestamp",
                ],
            ),
        ];

        for case in cases {
            let prom_expr = parse_metric_selector(case.0).unwrap();
            let table_provider = build_test_table_provider("some_metric".to_string(), 3, 3).await;
            let plan = PromPlanner::stmt_to_plan(table_provider, eval_stmt.clone())
                .await
                .unwrap();
            let mut fields = plan.schema().field_names();
            let mut expected = case.1.into_iter().map(String::from).collect::<Vec<_>>();
            fields.sort();
            expected.sort();
            assert_eq!(fields, expected, "case: {:?}", case.0);
        }

        let bad_cases = [
            r#"some_metric{__field__="nonexistent"}"#,
            r#"some_metric{__field__!="nonexistent"}"#,
        ];

        for case in bad_cases {
            let prom_expr = parser::parse(case).unwrap();
            eval_stmt.expr = prom_expr;
            let table_provider = build_test_table_provider("some_metric".to_string(), 3, 3).await;
            let plan = PromPlanner::stmt_to_plan(table_provider, eval_stmt.clone()).await;
            assert!(plan.is_err(), "case: {:?}", case);
        }
    }
}
