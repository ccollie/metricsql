use crate::eval::Evaluator;
use crate::search::join_tag_filter_list;
use crate::{Context, EvalConfig, QueryResults, QueryValue, RuntimeResult, SearchQuery, Timeseries};
use metricsql::ast::{DurationExpr, MetricExpr};
use metricsql::common::{LabelFilter, Value};
use std::sync::Arc;
use metricsql::prelude::ValueType;

/// An evaluator for a selector NOT containing a subquery or rollup
pub struct InstantVectorEvaluator {
    pub(crate) expr: MetricExpr,
    window: DurationExpr,
    offset: DurationExpr,
    tfs: Vec<Vec<LabelFilter>>,
}

impl InstantVectorEvaluator {
    #[inline]
    fn get_offset(&self, step: i64) -> i64 {
        self.offset.value(step)
    }

    #[inline]
    fn get_window(&self, step: i64) -> i64 {
        self.window.value(step)
    }

    pub(crate) fn search(
        &self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
    ) -> RuntimeResult<QueryResults> {
        let tfss = join_tag_filter_list(&self.tfs, &ec.enforced_tag_filters);
        let filters = tfss.to_vec();
        let sq = SearchQuery::new(ec.start, ec.end, filters, ec.max_series);
        ctx.process_search_query(&sq, &ec.deadline)
    }
}

impl Value for InstantVectorEvaluator {
    fn value_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}

impl Evaluator for InstantVectorEvaluator {
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let mut rss = self.search(ctx, &ec)?;

        let mut result: Vec<Timeseries> = Vec::with_capacity(rss.len());
        if rss.len() > 0 {
            for res in rss.series.iter_mut() {
                let timestamps = std::mem::take(&mut res.timestamps);
                let ts = Timeseries {
                    metric_name: std::mem::take(&mut res.metric_name),
                    values: std::mem::take(&mut res.values),
                    timestamps: Arc::new(timestamps),
                };
                result.push(ts);
            }
        }

        Ok(QueryValue::InstantVector(result))
    }

    fn return_type(&self) -> ValueType {
        ValueType::InstantVector
    }
}
