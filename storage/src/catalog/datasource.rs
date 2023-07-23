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
use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::datatypes::Fields;
use datafusion::catalog::catalog::CatalogProvider;
use datafusion::optimizer::utils::conjunction;
use datafusion::prelude::JoinType;
use datafusion::{
    arrow::{
        array::{Float64Array, Int64Array, StringArray},
        datatypes::Schema,
    },
    common::{OwnedTableReference, ScalarValue, TableReference},
    datasource::{DefaultTableSource, TableProvider},
    error::{DataFusionError, Result},
    logical_expr::{
        BinaryExpr, Expr as DfExpr, Extension, LogicalPlan, LogicalPlanBuilder, Operator,
    },
    prelude::{col, lit, Column, SessionContext},
};
use futures::future::try_join_all;
use metricsql::common::{LabelFilter, LabelFilterOp, MatchOp};
use metricsql::prelude::MetricExpr;
use runtime::{Label, RangeValue, Sample};
use snafu::{ensure, OptionExt, ResultExt};

use crate::error::{
    CatalogSnafu, ColumnNotFoundSnafu, DataFusionPlanningSnafu, TableNameNotFoundSnafu,
    TimeIndexNotFoundSnafu, UnknownTableSnafu, ValueNotFoundSnafu,
};
use crate::sql::extension_plan::{SeriesDivide, SeriesNormalize};
use crate::TableContext;

// https://github.com/splitgraph/seafowl/tree/main/datafusion_remote_tables

const DEFAULT_TIME_INDEX_COLUMN: &str = "time";

/// default value column name for empty metric
const DEFAULT_FIELD_COLUMN: &str = "value";

/// Special modifier to project field columns under multi-field mode
const FIELD_COLUMN_MATCHER: &str = "__field__";

#[derive(Default, Debug, Clone)]
struct PromPlannerContext {
    // query parameters
    start: Millisecond,
    end: Millisecond,

    // planner states
    time_index_column: Option<String>,
    table_names: Vec<String>,
    field_columns: Vec<String>,
    tag_columns: Vec<String>,
    name_matchers: Vec<Matcher>,
    field_column_matcher: Option<Vec<Matcher>>,
    /// The range in millisecond of range selector. None if there is no range selector.
    range: Option<Millisecond>,
    skip_nan: bool,
    supports_regex: bool,
}

pub struct SqlDataSource {
    ctx: PromPlannerContext,
    pub table_provider: Arc<Box<dyn TableProvider>>,
}

impl SqlDataSource {

    fn get_table(&self, table_name: &str) -> Result<Arc<dyn TableProvider>> {
        let table_ref = TableReference::from(table_name);
        let table = self
            .table_provider
            .resolve_table(table_ref)
            .context(CatalogSnafu)?;
        Ok(table)
    }

    // todo: generate datetime_filter
    #[tracing::instrument(name = "promql:engine:load_data", skip_all)]
    async fn selector_load_data(
        &mut self,
        selector: &MetricExpr,
        range: Option<Duration>,
    ) -> Result<()> {
        // https://promlabs.com/blog/2020/07/02/selecting-data-in-promql/#lookback-delta
        let start = self.ctx.start - range.map_or(self.ctx.lookback_delta, micros);
        let end = self.ctx.end; // 30 minutes + 5m = 35m

        // 1. Group by metrics (sets of label name-value pairs)
        let table_name = selector.metric_group;
        let filters: Vec<(&str, &str)> = selector
            .label_filters
            .iter()
            .filter(|mat| mat.op == LabelFilterOp::Equal)
            .map(|mat| (mat.label.as_str(), mat.value.as_str()))
            .collect();

        let ctxs = self
            .ctx
            .table_provider
            .create_context(table_name, (start, end), &filters)
            .await?;

        let mut tasks = Vec::new();
        for ctx in ctxs {
            let TableContext {
                session, schema, ..
            } = ctx;
            let selector = selector.clone();
            let task = tokio::task::spawn(async move {
                self.selector_load_data_from_datafusion(session, schema, selector, start, end)
                    .await
            });
            tasks.push(task);
            // update stats
            let mut ctx_scan_stats = self.ctx.scan_stats.write().await;
            ctx_scan_stats.add(&scan_stats);
        }
        let task_results = try_join_all(tasks)
            .await
            .map_err(|e| DataFusionError::Plan(format!("task error: {:?}", e)))?;

        let mut metrics: HashMap<String, RangeValue> = HashMap::default();
        let task_results_len = task_results.len();
        for task_result in task_results {
            if task_results_len == 1 {
                // only one ctx, no need to merge, just set it to metrics
                metrics = task_result?;
                break;
            }
            for (key, value) in task_result? {
                let metric = metrics
                    .entry(key)
                    .or_insert_with(|| RangeValue::new(value.labels, Vec::with_capacity(20)));
                metric.samples.extend(value.samples);
            }
        }

        // no data, return immediately
        if metrics.is_empty() {
            self.ctx
                .data_cache
                .write()
                .await
                .insert(table_name.to_string(), Value::None);
            return Ok(());
        }

        // cache data
        let mut metric_values = metrics.into_values().collect::<Vec<_>>();
        for metric in metric_values.iter_mut() {
            metric.samples.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        }
        let values = if metric_values.is_empty() {
            Value::None
        } else {
            Value::Matrix(metric_values)
        };
        self.ctx
            .data_cache
            .write()
            .await
            .insert(table_name.to_string(), values);
        Ok(())
    }

    async fn selector_load_data_from_datafusion(
        &self,
        ctx: TableContext,
        table_name: &str,
        selector: MetricExpr,
        start: i64,
        end: i64,
    ) -> Result<HashMap<u64, RangeValue>> {
        let table = match ctx.session.table(table_name).await {
            Ok(v) => v,
            Err(_) => {
                return Ok(HashMap::default());
            }
        };

        let timestamp_column = &ctx.timestamp_column;
        let value_column = &ctx.value_column;
        let skip_nan = self.ctx.skip_nan;

        let mut df_group = table.clone().filter(self.create_date_filter(start, end)?)?;
        for mat in selector.label_filters.iter() {
            if mat.name == timestamp_column
                || mat.name == value_column
                || ctx.schema.field_with_name(&mat.name).is_err()
            {
                continue;
            }
            df_group = df_group.filter(self.matcher_to_expr(mat, true)?)?;
        }

        if skip_nan {
            df_group = df_group.filter(col(value_column).is_not_null())?;
        }
        let batches = df_group
            .sort(vec![col(timestamp_column).sort(true, true)])?
            .collect()
            .await?;

        let mut metrics: HashMap<u64, RangeValue> = HashMap::default();

        for batch in &batches {
            let time_values = batch
                .column_by_name(timestamp_column)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            let value_values = batch
                .column_by_name(value_column)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            let mut labels_buf: Vec<Label> = Vec::with_capacity(batch.num_columns());
            let fields = batch.schema().fields();
            let columns = batch.columns();

            for i in 0..batch.num_rows() {
                labels_buf.clear();
                let hash = self.collect_row_labels(i, &mut labels_buf, fields, columns);
                let entry = metrics.entry(hash).or_insert_with(|| {
                    labels_buf.sort_by(|a, b| a.name.cmp(&b.name));
                    // maybe use an Arc to reduce memory usage
                    RangeValue::new(labels_buf.clone(), Vec::with_capacity(20))
                });
                entry
                    .samples
                    .push(Sample::new(time_values.value(i), value_values.value(i)));
            }
        }
        Ok(metrics)
    }

    fn collect_row_labels(
        &self,
        row: usize,
        labels: &mut Vec<Label>,
        fields: &Fields,
        columns: &[ArrayRef],
    ) -> u64 {
        let mut hasher = xxhash_rust::xxh3::Xxh3::new();
        for (k, v) in fields.iter().zip(columns) {
            let name = k.name();
            if name == self.timestamp_column || name == self.value_column {
                continue;
            }
            let value = v.as_any().downcast_ref::<StringArray>().unwrap();
            let str_value = value.value(row).to_string();

            hasher.write(name.as_bytes());
            hasher.write(str_value.as_bytes());

            labels.push(Label {
                name: name.to_string(),
                value: str_value,
            });
        }
        hasher.finish()
    }

    /// Extract metric name from `__name__` matcher and set it into [PromPlannerContext].
    /// Returns a new [Matchers] that doesn't contains metric name matcher.
    fn preprocess_label_matchers(&mut self, label_matchers: &Matchers) -> Result<Matchers> {
        let mut matchers = HashSet::new();
        for matcher in &label_matchers.matchers {
            if matcher.name == METRIC_NAME {
                if matches!(matcher.op, MatchOp::Equal) {
                    self.ctx.table_name = Some(matcher.value.clone());
                } else {
                    self.ctx
                        .field_column_matcher
                        .get_or_insert_default()
                        .push(matcher.clone());
                }
            } else {
                let _ = matchers.insert(matcher.clone());
            }
        }
        Ok(Matchers { matchers })
    }

    // filter_columns filters the columns in the table according to the field matchers.
    fn filter_columns(&self) {
        // make a projection plan if there is any `__field__` matcher
        if let Some(field_matchers) = &self.ctx.field_column_matcher {
            let col_set = self.ctx.field_columns.iter().collect::<HashSet<_>>();
            // opt-in set
            let mut result_set = HashSet::new();
            // opt-out set
            let mut reverse_set = HashSet::new();
            for matcher in field_matchers {
                match &matcher.op {
                    MatchOp::Equal => {
                        if col_set.contains(&matcher.value) {
                            let _ = result_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                                location: Default::default(),
                            }
                            .build());
                        }
                    }
                    MatchOp::NotEqual => {
                        if col_set.contains(&matcher.value) {
                            let _ = reverse_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                            }
                            .build());
                        }
                    }
                    MatchOp::Re(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
                                let _ = result_set.insert(col.clone());
                            }
                        }
                    }
                    MatchOp::NotRe(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
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

            self.ctx.field_columns = result_set.iter().cloned().collect();
        }
    }

    // filter_columns filters the columns in the table according to the field matchers.
    fn extract_metrics(&self) {
        // make a projection plan if there is any `__name__` matcher
        if let Some(field_matchers) = &self.ctx.field_column_matcher {
            let col_set = self.ctx.field_columns.iter().collect::<HashSet<_>>();
            // opt-in set
            let mut result_set = HashSet::new();
            // opt-out set
            let mut reverse_set = HashSet::new();
            for matcher in field_matchers {
                match &matcher.op {
                    MatchOp::Equal => {
                        if col_set.contains(&matcher.value) {
                            let _ = result_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                                location: Default::default(),
                            }
                                .build());
                        }
                    }
                    MatchOp::NotEqual => {
                        if col_set.contains(&matcher.value) {
                            let _ = reverse_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                            }
                                .build());
                        }
                    }
                    MatchOp::Re(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
                                let _ = result_set.insert(col.clone());
                            }
                        }
                    }
                    MatchOp::NotRe(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
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

            self.ctx.field_columns = result_set.iter().cloned().collect();
        }
    }

    // TODO(ruihang): ignore `MetricNameLabel` (`__name__`) matcher
    fn matcher_to_expr(&self, matcher: &LabelFilter, supports_regex: bool) -> Result<DfExpr> {
        use LabelFilterOp::*;
        let col = DfExpr::Column(Column::from_name(matcher.name));
        let lit = DfExpr::Literal(ScalarValue::Utf8(Some(matcher.value)));
        let expr = match matcher.op {
            Equal => col.eq(lit),
            NotEqual => col.not_eq(lit),
            MatchOp::Re(_) => {
                if !supports_regex {
                    let regexp_match_udf = crate::udf::regex_match_udf().clone();
                    regexp_match_udf.call(vec![col, lit])?
                } else {
                    DfExpr::BinaryExpr(BinaryExpr {
                        left: Box::new(col),
                        op: Operator::RegexMatch,
                        right: Box::new(lit),
                    })
                }
            }
            MatchOp::NotRe(_) => {
                if !supports_regex {
                    let regexp_not_match_udf = crate::udf::regex_not_match_udf().clone();
                    regexp_not_match_udf.call(vec![col, lit])?
                } else {
                    DfExpr::BinaryExpr(BinaryExpr {
                        left: Box::new(col),
                        op: Operator::RegexNotMatch,
                        right: Box::new(lit),
                    })
                }
            }
        };

        Ok(expr)
    }

    // TODO(ruihang): ignore `MetricNameLabel` (`__name__`) matcher
    fn matchers_to_expr(
        &self,
        label_matchers: &[LabelFilter],
        in_memory: bool,
    ) -> Result<Vec<DfExpr>> {
        label_matchers
            .iter()
            .map(|matcher| self.matcher_to_expr(matcher, in_memory))
            .collect::<Result<_>>()
    }

    fn create_date_filter(&self, start: i64, end: i64) -> Result<DfExpr> {
        self.create_time_index_column_expr()?
            .gt_eq(DfExpr::Literal(ScalarValue::TimestampMillisecond(
                Some(start),
                None,
            )))
            .and(self.create_time_index_column_expr()?.lt_eq(DfExpr::Literal(
                ScalarValue::TimestampMillisecond(Some(end), None),
            )))
    }

    async fn selector_to_series_normalize_plan(
        &mut self,
        label_matchers: Matchers,
        is_range_selector: bool,
    ) -> Result<LogicalPlan> {
        let table_name = self.ctx.table_name.clone().unwrap();

        // make filter exprs
        let range_ms = self.ctx.range.unwrap_or_default();
        let mut scan_filters = self.matchers_to_expr(label_matchers.clone())?;
        scan_filters.push(self.create_time_index_column_expr()?.gt_eq(DfExpr::Literal(
            ScalarValue::TimestampMillisecond(Some(self.ctx.start), None),
        )));
        scan_filters.push(self.create_time_index_column_expr()?.lt_eq(DfExpr::Literal(
            ScalarValue::TimestampMillisecond(Some(self.ctx.end), None),
        )));

        // make table scan with filter exprs
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
                    MatchOp::Equal => {
                        if col_set.contains(&matcher.value) {
                            let _ = result_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                                location: Default::default(),
                            }
                            .build());
                        }
                    }
                    MatchOp::NotEqual => {
                        if col_set.contains(&matcher.value) {
                            let _ = reverse_set.insert(matcher.value.clone());
                        } else {
                            return Err(ColumnNotFoundSnafu {
                                col: matcher.value.clone(),
                            }
                            .build());
                        }
                    }
                    MatchOp::Re(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
                                let _ = result_set.insert(col.clone());
                            }
                        }
                    }
                    MatchOp::NotRe(regex) => {
                        for col in &self.ctx.field_columns {
                            if regex.is_match(col) {
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

            self.ctx.field_columns = result_set.iter().cloned().collect();
            let exprs = result_set
                .into_iter()
                .map(|col| DfExpr::Column(col.into()))
                .chain(self.create_tag_column_exprs()?.into_iter())
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
        let accurate_filters = self.matchers_to_expr(label_matchers)?;
        if !accurate_filters.is_empty() {
            plan_builder = plan_builder
                .filter(conjunction(accurate_filters).unwrap())
                .context(DataFusionPlanningSnafu)?;
        }
        let sort_plan = plan_builder
            .sort(self.create_tag_and_time_index_column_sort_exprs()?)
            .context(DataFusionPlanningSnafu)?
            .build()
            .context(DataFusionPlanningSnafu)?;

        // make divide plan
        let divide_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(SeriesDivide::new(self.ctx.tag_columns.clone(), sort_plan)),
        });

        // make series_normalize plan
        let series_normalize = SeriesNormalize::new(
            offset_duration,
            self.ctx
                .time_index_column
                .clone()
                .with_context(|| TimeIndexNotFoundSnafu { table: table_name })?,
            is_range_selector,
            divide_plan,
        );
        let logical_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(series_normalize),
        });

        Ok(logical_plan)
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
            .context(UnknownTableSnafu)?
            .table();

        // set time index column name
        let time_index = table
            .schema()
            .timestamp_column()
            .with_context(|| TimeIndexNotFoundSnafu { table: table_name })?
            .name
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

    /// Build a projection that projects and performs operation expr for all value columns.
    /// Non-value columns (tag and timestamp) will be preserved in the projection.
    ///
    /// # Side effect
    ///
    /// This function will update the value columns in the context. Those new column names
    /// don't contains qualifier.
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

        // build computation exprs
        let result_field_columns = self
            .ctx
            .field_columns
            .iter()
            .map(name_to_expr)
            .collect::<Result<Vec<_>>>()?;

        // alias the computation exprs to remove qualifier
        self.ctx.field_columns = result_field_columns
            .iter()
            .map(|expr| expr.display_name())
            .collect::<DfResult<Vec<_>>>()
            .context(DataFusionPlanningSnafu)?;
        let field_columns_iter = result_field_columns
            .into_iter()
            .zip(self.ctx.field_columns.iter())
            .map(|(expr, name)| Ok(DfExpr::Alias(Box::new(expr), name.to_string())));

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
