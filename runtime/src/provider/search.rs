use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use metricsql::common::{LabelFilter, Matchers};

use crate::execution::Context;
use crate::functions::remove_nan_values_in_place;
use crate::provider::Deadline;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{MetricName, Timeseries, Timestamp, TimestampTrait};

pub type TimeRange = Range<Timestamp>;

// todo: async ???. Add context ?
pub trait MetricDataProvider: Sync + Send {
    fn search(&self, sq: &SearchQuery, deadline: &Deadline) -> RuntimeResult<QueryResults>;
}

pub struct NullMetricDataProvider {}

impl MetricDataProvider for NullMetricDataProvider {
    fn search(&self, _sq: &SearchQuery, _deadline: &Deadline) -> RuntimeResult<QueryResults> {
        let qr = QueryResults::default();
        Ok(qr)
    }
}

/// QueryableFunc is an adapter to allow the use of ordinary functions as
/// MetricDataProvider. It follows the idea of http.HandlerFunc.
pub type QueryableFunc = fn(ctx: &Context, sq: &SearchQuery) -> RuntimeResult<QueryResults>;

/// SearchQuery is used for sending provider queries to external data sources.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The time range for searching time series
    /// TODO: use TimestampRange
    pub start: Timestamp,
    pub end: Timestamp,

    /// Tag filters for the provider query
    pub matchers: Vec<Matchers>,

    /// The maximum number of time series the provider query can return.
    pub max_metrics: usize,
}

impl SearchQuery {
    /// Create a new provider query for the given args.
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        tag_filter_list: Vec<Matchers>,
        max_metrics: usize,
    ) -> Self {
        let mut max = max_metrics;
        if max_metrics == 0 {
            max = 2e9 as usize
        }
        SearchQuery {
            start,
            end,
            matchers: tag_filter_list,
            max_metrics: max,
        }
    }
}

impl Display for SearchQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut a: Vec<String> = Vec::with_capacity(self.matchers.len());
        for tfs in &self.matchers {
            a.push(filters_to_string(tfs))
        }
        let start = self.start.to_string_millis();
        let end = self.end.to_string_millis();
        write!(
            f,
            "filters={}, timeRange=[{}..{}], max={}",
            a.join(","),
            start,
            end,
            self.max_metrics
        )?;

        Ok(())
    }
}

fn filters_to_string(tfs: &[LabelFilter]) -> String {
    let mut a: Vec<String> = Vec::with_capacity(tfs.len() + 2);
    a.push("{{".to_string());
    for tf in tfs {
        a.push(format!("{}", tf))
    }
    a.push("}}".to_string());
    a.join(",")
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
    pub(crate) rows_processed: usize,
    /// Internal only
    pub(crate) worker_id: u64,
}

impl QueryResult {
    pub fn new() -> Self {
        QueryResult {
            metric: MetricName::default(),
            values: vec![],
            timestamps: vec![],
            rows_processed: 0,
            worker_id: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        QueryResult {
            metric: MetricName::default(),
            values: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
            rows_processed: 0,
            worker_id: 0,
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
#[derive(Debug)]
pub struct QueryResults {
    pub series: Vec<QueryResult>,
    signal: Arc<AtomicU32>,
}

impl Default for QueryResults {
    fn default() -> Self {
        Self {
            series: vec![],
            signal: Arc::new(AtomicU32::new(0_u32)),
        }
    }
}

impl Clone for QueryResults {
    fn clone(&self) -> Self {
        let signal_value = self.signal.load(Ordering::Relaxed);
        QueryResults {
            series: self.series.clone(),
            signal: Arc::new(AtomicU32::new(signal_value)),
        }
    }
}

impl QueryResults {
    pub(crate) fn new(series: Vec<QueryResult>) -> Self {
        QueryResults {
            series,
            signal: Arc::new(AtomicU32::new(0_u32)),
        }
    }

    /// Len returns the number of results in rss.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    pub fn should_stop(&mut self) -> bool {
        if self.signal.fetch_add(0, Ordering::Relaxed) != 0 {
            return true;
        }
        false
    }

    pub fn is_cancelled(&self) -> bool {
        self.signal.fetch_add(0, Ordering::Relaxed) == 1
    }

    /// run_parallel runs f in parallel for all the results from rss.
    ///
    /// f shouldn't hold references to rs after returning.
    /// Data processing is immediately stopped if f returns an error.
    ///
    /// rss becomes unusable after the call to run_parallel.
    pub(crate) fn run_parallel<C: Sync + Send, F>(&mut self, ctx: &mut C, f: F) -> RuntimeResult<()>
    where
        F: Fn(Arc<&mut C>, &mut QueryResult, u64) -> RuntimeResult<()> + Send + Sync,
    {
        for (id, ts) in self.series.iter_mut().enumerate() {
            ts.worker_id = id as u64;
        }

        let must_stop = AtomicBool::new(false);
        let sharable_ctx = Arc::new(ctx);

        self.series
            .par_iter_mut()
            .filter(|rs| !rs.timestamps.is_empty())
            .try_for_each(|rs| {
                if must_stop.load(Ordering::Relaxed) {
                    return Err(RuntimeError::TaskCancelledError("Search".to_string()));
                }

                let worker_id = rs.worker_id;
                f(Arc::clone(&sharable_ctx), rs, worker_id)
            })
    }

    pub fn cancel(&self) {
        self.signal.store(1, Ordering::Relaxed);
    }
}

pub fn remove_empty_values_and_timeseries(tss: &mut Vec<QueryResult>) {
    tss.retain_mut(|ts| {
        remove_nan_values_in_place(&mut ts.values, &mut ts.timestamps);
        !ts.values.is_empty()
    });
}
