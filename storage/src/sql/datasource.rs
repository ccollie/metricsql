use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use datafusion::{
    arrow::{
        array::{Float64Array, Int64Array, StringArray},
        datatypes::Schema,
    },
    error::{DataFusionError, Result},
    prelude::{col, lit, SessionContext},
};
use datafusion::catalog::catalog::CatalogProvider;
use datafusion::datasource::TableProvider;

use metricsql::common::LabelFilterOp;
use metricsql::prelude::MetricExpr;

pub struct SqlDataSource {
    pub catalog: Arc<Box<dyn CatalogProvider>>,
    pub table_provider: Arc<Box<dyn TableProvider>>,
    pub timestamp_column: String,
}

impl SqlDataSource {
    fn map_to_table_name(metric_name: &str) -> String {
        // todo: map to field instead
        metric_name.to_string()
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
            .create_context(&self.ctx.org_id, table_name, (start, end), &filters)
            .await?;

        let mut tasks = Vec::new();
        for (ctx, schema, scan_stats) in ctxs {
            let selector = selector.clone();
            let task = tokio::task::spawn(async move {
                self.selector_load_data_from_datafusion(ctx,
                                                        schema,
                                                        selector,
                                                        start,
                                                        end).await
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

    fn generate_datetime_filter() {}

    async fn selector_load_data_from_datafusion(
        &self,
        ctx: SessionContext,
        schema: Arc<Schema>,
        selector: MetricExpr,
        start: i64,
        end: i64,
    ) -> Result<HashMap<String, RangeValue>> {
        let table_name = selector.metric_group;
        let table = match ctx.table(table_name).await {
            Ok(v) => v,
            Err(_) => {
                return Ok(HashMap::default());
            }
        };

        let mut df_group = table.clone().filter(
            col(&self.timestamp_column)
                .gt(lit(start))
                .and(col(&self.timestamp_column).lt_eq(lit(end))),
        )?;
        for mat in selector.label_filters.iter() {
            if mat.name == self.timestamp_column
                || mat.name == VALUE_LABEL
                || schema.field_with_name(&mat.name).is_err()
            {
                continue;
            }
            match &mat.op {
                MatchOp::Equal => {
                    df_group = df_group.filter(col(mat.name.clone()).eq(lit(mat.value.clone())))?
                }
                MatchOp::NotEqual => {
                    df_group = df_group.filter(col(mat.name.clone()).not_eq(lit(mat.value.clone())))?
                }
                MatchOp::Re(_re) => {
                    let regexp_match_udf =
                        crate::service::search::datafusion::regexp_udf::REGEX_MATCH_UDF.clone();
                    df_group = df_group.filter(
                        regexp_match_udf.call(vec![col(mat.name.clone()), lit(mat.value.clone())]),
                    )?
                }
                MatchOp::NotRe(_re) => {
                    let regexp_not_match_udf =
                        crate::service::search::datafusion::regexp_udf::REGEX_NOT_MATCH_UDF.clone();
                    df_group = df_group.filter(
                        regexp_not_match_udf.call(vec![col(mat.name.clone()), lit(mat.value.clone())]),
                    )?
                }
            }
        }
        let batches = df_group
            .sort(vec![col(&self.timestamp_column).sort(true, true)])?
            .collect()
            .await?;

        let mut metrics: HashMap<String, RangeValue> = HashMap::default();
        for batch in &batches {
            let hash_values = batch
                .column_by_name(HASH_LABEL)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let time_values = batch
                .column_by_name(timestamp_column)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let value_values = batch
                .column_by_name(VALUE_LABEL)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let hash = hash_values.value(i).to_string();
                let entry = metrics.entry(hash).or_insert_with(|| {
                    let mut labels = Vec::with_capacity(batch.num_columns());
                    for (k, v) in batch.schema().fields().iter().zip(batch.columns()) {
                        let name = k.name();
                        if name == timestamp_column
                            || name == HASH_LABEL
                            || name == VALUE_LABEL
                        {
                            continue;
                        }
                        let value = v.as_any().downcast_ref::<StringArray>().unwrap();
                        labels.push(Arc::new(Label {
                            name: name.to_string(),
                            value: value.value(i).to_string(),
                        }));
                    }
                    labels.sort_by(|a, b| a.name.cmp(&b.name));
                    RangeValue::new(labels, Vec::with_capacity(20))
                });
                entry
                    .samples
                    .push(Sample::new(time_values.value(i), value_values.value(i)));
            }
        }
        Ok(metrics)
    }
}


