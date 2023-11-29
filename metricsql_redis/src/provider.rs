use redis::{Client, Commands, Connection};
use redis_ts::{AsyncCommands, TsCommands, TsFilterOptions, TsRangeQuery};

use futures::future::join_all;
use metricsql_engine::prelude::TimestampTrait;
use metricsql_engine::provider::MetricDataProvider;
use metricsql_engine::runtime_error::{RuntimeError, RuntimeResult};
use metricsql_engine::{Deadline, MetricName, QueryResult, QueryResults, SearchQuery};
use metricsql_parser::label::{LabelFilterOp, Matchers};

pub struct RedisMetricsQLProvider {
    connection: Connection,
    client: Client,
    pub url: String,
}

impl RedisMetricsQLProvider {
    pub fn new(url: &str) -> RuntimeResult<Self> {
        let client = Client::open(url).map_err(|e| {
            RuntimeError::ProviderError(format!("Failed to open redis client: {:?}", e))
        })?;

        let connection = client.get_connection().map_err(|e| {
            RuntimeError::ProviderError(format!("Failed to get redis connection: {:?}", e))
        })?;

        Ok(Self {
            client,
            connection,
            url: url.to_string(),
        })
    }

    fn get_metric_names(&mut self) -> RuntimeResult<Vec<MetricName>> {
        let mut metric_names = Vec::new();
        let keys: Vec<String> = self.connection.keys("*").unwrap();
        for key in keys {
            metric_names.push(MetricName::from(key));
        }
        Ok(metric_names)
    }

    async fn fetch_data(
        &mut self,
        sq: &SearchQuery,
        deadline: &Deadline,
    ) -> RuntimeResult<QueryResults> {
        let mut qr = QueryResults::default();
        let mut filter_options = TsFilterOptions::default();
        filter_options.with_labels(true);

        let mut range_query = TsRangeQuery::default();
        range_query.from(sq.start);
        range_query.to(sq.end);

        let calls = Vec::with_capacity(sq.matchers.len());
        for matcher in sq.matchers.iter() {}
        let calls = sq
            .matchers
            .iter()
            .map(|matcher| append_filter(&mut filter_options, matcher));

        Ok(qr)
    }

    async fn fetch_one(
        mut self,
        start: i64,
        end: i64,
        matchers: &Matchers,
    ) -> RuntimeResult<Vec<QueryResult>> {
        let mut range_query = TsRangeQuery::default();
        range_query.from(start);
        range_query.to(end);

        let mut filter_options = TsFilterOptions::default();
        filter_options.with_labels(true);
        append_filter(&mut filter_options, matchers)?;

        let from_redis = self
            .connection
            .ts_mrange(range_query, filter_options)
            .map_err(|e| {
                RuntimeError::ProviderError(format!("Failed to fetch data from redis: {:?}", e))
            })?;

        let mut result = Vec::with_capacity(from_redis.values.len());
        for entry in from_redis.values.iter() {
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
                query_result.timestamps.push(*ts);
            }

            result.push(query_result);
        }

        Ok(result)
    }
}

impl MetricDataProvider for RedisMetricsQLProvider {
    fn search(&self, sq: &SearchQuery, deadline: &Deadline) -> RuntimeResult<QueryResults> {
        let qr = QueryResults::default();
        let mut filter_options = TsFilterOptions::default();
        filter_options.with_labels(true);

        let mut range_query = TsRangeQuery::default();
        range_query.from(sq.start);
        range_query.to(sq.end);

        Ok(qr)
    }
}

fn append_filter(filter_options: &mut TsFilterOptions, matchers: &Matchers) -> RuntimeResult<()> {
    for label_filter in matchers.iter() {
        match label_filter.op {
            LabelFilterOp::Equal => {
                filter_options.equals(label_filter.label.as_str(), label_filter.value.as_str());
            }
            LabelFilterOp::NotEqual => {
                filter_options.not_equals(label_filter.label.as_str(), label_filter.value.as_str());
            }
            LabelFilterOp::RegexEqual => {
                let or_values = get_or_values_from_regex(label_filter.value.as_str())?;
                filter_options.not_in_set(label_filter.label.as_str(), or_values);
            }
            LabelFilterOp::RegexNotEqual => {
                let or_values = get_or_values_from_regex(label_filter.value.as_str())?;
                filter_options.not_in_set(label_filter.label.as_str(), or_values);
            }
        }
    }
    Ok(())
}

/// RedisTimeseries does not support regexes. We attempt here to translate certain regexes into
/// a form that regex can handle - specifically alternations.
fn get_or_values_from_regex(pattern: &str) -> RuntimeResult<Vec<String>> {
    todo!()
}
