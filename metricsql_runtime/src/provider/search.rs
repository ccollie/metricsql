use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;

use metricsql_parser::label::Matchers;

use crate::execution::Context;
use crate::functions::remove_nan_values_in_place;
use crate::provider::Deadline;
use crate::runtime_error::RuntimeResult;
use crate::types::{MetricName, Timeseries, Timestamp, TimestampTrait};

pub type TimeRange = Range<Timestamp>;


#[async_trait]
pub trait MetricStorage: Sync + Send {
    async fn search(&self, sq: SearchQuery, deadline: Deadline) -> RuntimeResult<QueryResults>;
}

pub struct NullMetricStorage {}

#[async_trait]
impl MetricStorage for NullMetricStorage {
    async fn search(&self, _sq: SearchQuery, _deadline: Deadline) -> RuntimeResult<QueryResults> {
        let qr = QueryResults::default();
        Ok(qr)
    }
}

/// QueryableFunc is an adapter to allow the use of ordinary functions as
/// MetricDataProvider. It follows the idea of http.HandlerFunc.
pub type QueryableFunc = fn(ctx: &Context, sq: SearchQuery) -> RuntimeResult<QueryResults>;

/// SearchQuery is used for sending provider queries to external data sources.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The time range for searching time series
    /// TODO: use TimestampRange
    pub start: Timestamp,
    pub end: Timestamp,

    /// Tag filters for the provider query
    pub matchers: Matchers,

    /// The maximum number of time series the provider query can return.
    pub max_metrics: usize,
}

impl SearchQuery {
    /// Create a new provider query for the given args.
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        matchers: Matchers,
        max_metrics: usize,
    ) -> Self {
        let mut max = max_metrics;
        if max_metrics == 0 {
            max = 2e9 as usize
        }
        let start = if start < 0 {
            0
        } else {
            start
        };
        SearchQuery {
            start,
            end,
            matchers,
            max_metrics: max,
        }
    }
}

impl Display for SearchQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start = self.start.to_string_millis();
        let end = self.end.to_string_millis();
        write!(
            f,
            "filters={}, timeRange=[{}..{}], max={}", self.matchers, start, end, self.max_metrics
        )?;

        Ok(())
    }
}

/// InstantQuery is used for returning instant query results from external data sources.
#[derive(Debug, Clone)]
pub struct InstantQueryResult {
    pub metrics: Vec<MetricName>,
    pub timestamp: i64,
    pub values: Vec<f64>,
}

/// RangeQueryResult is a single timeseries result returned from a range query.
///
/// ProcessSearchQuery returns QueryResult slices.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct RangeQueryResult {
    /// The name of the metric.
    pub metric: MetricName,
    /// Values are sorted by Timestamps.
    pub values: Vec<f64>,
    pub timestamps: Vec<i64>,
}

/// QueryResult is a single timeseries result.
///
/// ProcessSearchQuery returns QueryResult slices.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// The name of the metric.
    pub metric: MetricName,
    /// Values are sorted by Timestamps.
    pub values: Vec<f64>,
    pub timestamps: Vec<i64>,
}

impl QueryResult {
    pub fn new(metric: MetricName, timestamps: Vec<i64>, values: Vec<f64>) -> Self {
        QueryResult {
            metric,
            values,
            timestamps,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        QueryResult {
            metric: MetricName::default(),
            values: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
        }
    }

    pub fn into_timeseries(mut self) -> Timeseries {
        Timeseries {
            metric_name: std::mem::take(&mut self.metric),
            values: std::mem::take(&mut self.values),
            timestamps: Arc::new(std::mem::take(&mut self.timestamps)),
        }
    }

    pub fn reset(&mut self) {
        self.metric.reset();
        self.values.clear();
        self.timestamps.clear();
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

/// Results holds results returned from ProcessSearchQuery.
#[derive(Default, Debug, Clone)]
pub struct QueryResults {
    pub series: Vec<QueryResult>,
}

impl QueryResults {
    pub fn new(series: Vec<QueryResult>) -> Self {
        QueryResults { series }
    }

    /// Len returns the number of results in rss.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
}

pub fn remove_empty_values_and_timeseries(tss: &mut Vec<QueryResult>) {
    tss.retain_mut(|ts| {
        remove_nan_values_in_place(&mut ts.values, &mut ts.timestamps);
        !ts.values.is_empty()
    });
}
