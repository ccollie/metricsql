use std::borrow::Cow;

use async_trait::async_trait;
use futures::future::try_join_all;
use rustis::commands::TsRangeSample;
use rustis::{
    client::Client,
    commands::{TimeSeriesCommands, TsGroupByOptions, TsMRangeOptions},
};

use metricsql_common::regex_util::get_or_values;
use metricsql_engine::provider::MetricStorageProvider;
use metricsql_engine::runtime_error::{RuntimeError, RuntimeResult};
use metricsql_engine::{Deadline, MetricName, QueryResult, QueryResults, SearchQuery};
use metricsql_parser::label::{LabelFilterOp, Matchers, NAME_LABEL};

/// Trait to get the metric label name from label matchers
pub trait MetricLabelMapper: Send + Sync {
    fn get_label(&self, matchers: &Matchers) -> Cow<String>;
}

pub struct DefaultMetricLabelMapper {
    label: String,
}

impl DefaultMetricLabelMapper {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
        }
    }
}

impl Default for DefaultMetricLabelMapper {
    fn default() -> Self {
        Self {
            label: NAME_LABEL.to_string(),
        }
    }
}

impl MetricLabelMapper for DefaultMetricLabelMapper {
    fn get_label(&self, _matchers: &Matchers) -> Cow<String> {
        Cow::Borrowed(&self.label)
    }
}

pub struct RedisMetricsQLProvider {
    pub client: Client,
    pub url: String,
    pub label_name_mapper: Box<dyn MetricLabelMapper>,
}

impl RedisMetricsQLProvider {
    pub async fn new(url: &str) -> RuntimeResult<Self> {
        let client = Client::connect(url).await.map_err(|e| {
            RuntimeError::ProviderError(format!("Failed to open redis client: {:?}", e))
        })?;

        Ok(Self {
            client,
            url: url.to_string(),
            label_name_mapper: Box::new(DefaultMetricLabelMapper::default()),
        })
    }

    pub fn set_label_name(&mut self, label: &str) {
        self.label_name_mapper = Box::new(DefaultMetricLabelMapper::new(label));
    }

    pub fn set_label_name_mapper(&mut self, mapper: Box<dyn MetricLabelMapper>) {
        self.label_name_mapper = mapper;
    }

    async fn fetch_data(
        &self,
        sq: SearchQuery,
        _deadline: Deadline,
    ) -> RuntimeResult<QueryResults> {
        let mut calls = Vec::with_capacity(sq.matchers.len());

        for matcher in sq.matchers.iter() {
            calls.push(self.fetch_one(sq.start, sq.end, matcher));
        }

        let results = try_join_all(calls).await.map_err(|e| {
            RuntimeError::ProviderError(format!("Failed to fetch data from redis: {:?}", e))
        })?;

        let mut qr = QueryResults::default();
        for result in results {
            qr.series.extend(result);
        }
        Ok(qr)
    }

    async fn fetch_one(
        &self,
        start: i64,
        end: i64,
        matchers: &Matchers,
    ) -> RuntimeResult<Vec<QueryResult>> {
        let range_options = TsMRangeOptions::default().withlabels();
        let mut filters = Vec::with_capacity(matchers.len());
        self.append_filter(&mut filters, matchers)?;

        let from_redis: Vec<TsRangeSample> = self
            .client
            .ts_mrange(
                start,
                end,
                range_options,
                filters,
                TsGroupByOptions::default(),
            )
            .await
            .map_err(|e| {
                RuntimeError::ProviderError(format!("Failed to fetch data from redis: {:?}", e))
            })?;

        let mut result = Vec::with_capacity(from_redis.len());
        for entry in from_redis.iter() {
            let mut metric_name = MetricName::default();
            for (key, value) in entry.labels.iter() {
                metric_name.set_tag(key, value);
            }
            let mut query_result = QueryResult::default();
            query_result.metric = metric_name;
            query_result.values.reserve(entry.values.len());
            query_result.timestamps.reserve(entry.values.len());

            for (ts, v) in entry.values.iter() {
                query_result.values.push(*v);
                query_result.timestamps.push(*ts as i64);
            }

            result.push(query_result);
        }

        Ok(result)
    }

    fn append_filter(
        &self,
        filter_options: &mut Vec<String>,
        matchers: &Matchers,
    ) -> RuntimeResult<()> {
        for label_filter in matchers.iter() {
            let label = if label_filter.label == NAME_LABEL {
                self.label_name_mapper.get_label(matchers)
            } else {
                Cow::Borrowed(&label_filter.label)
            };
            match label_filter.op {
                LabelFilterOp::Equal => {
                    filter_options.push(format!("{}={}", label, label_filter.value));
                }
                LabelFilterOp::NotEqual => {
                    filter_options.push(format!("{}!={}", label, label_filter.value));
                }
                LabelFilterOp::RegexEqual => {
                    let or_values = get_or_values_from_regex(label_filter.value.as_str())?;
                    filter_options.push(format!("{}=({})", label, or_values.join(",")));
                }
                LabelFilterOp::RegexNotEqual => {
                    let or_values = get_or_values_from_regex(label_filter.value.as_str())?;
                    filter_options.push(format!("{}!=({})", label, or_values.join(",")));
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl MetricStorageProvider for RedisMetricsQLProvider {
    async fn search(&self, sq: SearchQuery, deadline: Deadline) -> RuntimeResult<QueryResults> {
        self.fetch_data(sq, deadline).await
    }
}

/// RedisTimeseries does not support regexes. We attempt here to translate certain regexes into
/// a form that redis can handle - specifically alternations.
fn get_or_values_from_regex(pattern: &str) -> RuntimeResult<Vec<String>> {
    let alternates = get_or_values(pattern);
    if alternates.is_empty() {
        // todo: specific Unsupported variant
        return Err(RuntimeError::ProviderError(
            "redis provider: unsupported regex".into(),
        ));
    }
    Ok(alternates)
}
