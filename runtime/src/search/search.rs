use lockfree_object_pool::{LinearObjectPool, LinearReusable};
use once_cell::sync::OnceCell;
use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::{Arc};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use metricsql::ast::LabelFilter;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::{MetricName, Timeseries};
use crate::traits::{Timestamp, TimestampTrait};

pub type TimeRange = Range<Timestamp>;

/// Search is a search for time series.
#[derive(Debug, Clone, Default)]
pub struct Search {
    /// tr contains time range used in the search.
    pub tr: TimeRange,

    /// tfss contains tag filters used in the search.
    pub tfss: Vec<LabelFilter>,

    /// deadline in unix timestamp seconds for the current search.
    pub deadline: u64, // todo: Duration

    need_closing: bool,
}

impl Search {
    pub fn reset(&mut self) {
        self.tr = TimeRange{ start: Timestamp::now(), end: i64::MAX };
        self.tfss = vec![];
        self.deadline = 0;
        self.need_closing = false;
    }
}

/// SearchQuery is used for sending search queries to external data sources.
#[derive(Default, Debug, Clone)]
pub struct SearchQuery {
    /// The time range for searching time series
    pub min_timestamp: Timestamp,
    pub max_timestamp: Timestamp,

    /// Tag filters for the search query
    pub tag_filterss: Vec<Vec<LabelFilter>>,

    /// The maximum number of time series the search query can return.
    pub max_metrics: usize
}

impl SearchQuery {
    /// Create a new search query for the given args.
    pub fn new(start: Timestamp, end: Timestamp, tag_filterss: Vec<Vec<LabelFilter>>, max_metrics: usize) -> Self {
        let mut max = max_metrics;
        if max_metrics <= 0 {
            max = 2e9 as usize
        }
        SearchQuery{
            min_timestamp: start,
            max_timestamp: end,
            tag_filterss,
            max_metrics: max,
        }
    }
}

impl Display for SearchQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut a: Vec<String> = Vec::with_capacity(self.tag_filterss.len());
        for tfs in &self.tag_filterss {
            a.push(filters_to_string(&tfs))
        }
        let start = self.min_timestamp.to_string_millis();
        let end = self.max_timestamp.to_string_millis();
        write!(f, "filters={}, timeRange=[{}..{}], max={}", a.join(","), start, end, self.max_metrics)?;

        Ok(())
    }
}

fn filters_to_string(tfs: &[LabelFilter]) -> String {
    let mut a: Vec<String> = Vec::with_capacity(tfs.len() + 2);
    a.push("{{".to_string());
    for tf in tfs {
        a.push(format!("{}", tf) )
    }
    a.push("}}".to_string());
    return a.join( ",");
}

/// QueryResult is a single timeseries result.
///
/// ProcessSearchQuery returns QueryResult slices.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// The name of the metric.
    pub metric_name: MetricName,
    /// Values are sorted by Timestamps.
    pub values: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub(crate) rows_processed: usize,
    /// Internal only
    pub(crate) worker_id: u64,

    pub(crate) last_reset_time: Timestamp
}

impl QueryResult {
    pub fn new() -> Self {
        QueryResult {
            metric_name: MetricName::default(),
            values: vec![],
            timestamps: vec![],
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        QueryResult {
            metric_name: MetricName::default(),
            values: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
            rows_processed: 0,
            worker_id: 0,
            last_reset_time: 0
        }
    }

    pub fn into_timeseries(mut self) -> Timeseries {
        Timeseries {
            metric_name: std::mem::take(&mut self.metric_name),
            values: std::mem::take(&mut self.values),
            timestamps: Arc::new(std::mem::take(&mut self.timestamps))
        }
    }

    pub fn reset(&mut self) {
        self.metric_name.reset();
        self.values.clear();
        self.timestamps.clear();
    }

    fn shrink_to_fit(&mut self) {
        self.metric_name.reset();
        self.values.shrink_to_fit();
    }

    fn shrink(&mut self, min_capacity: usize) {
        self.values.shrink_to(min_capacity);
        self.timestamps.shrink_to(min_capacity);
    }

    pub fn len(self) -> usize {
        self.timestamps.len()
    }
}


/// Results holds results returned from ProcessSearchQuery.
#[derive(Debug)]
pub struct QueryResults {
    pub tr: TimeRange,
    pub deadline: Deadline,
    pub series: Vec<QueryResult>,
    pub sr: Search,
    signal: Arc<AtomicU32>
}

pub(crate) type SearchParallelFn = fn(rs: &mut QueryResult, worker_id: u64) -> RuntimeResult<()>;

impl Default for QueryResults {
    fn default() -> Self {
        Self {
            tr: TimeRange::default(),
            deadline: Deadline::default(),
            series: vec![],
            sr: Search::default(),
            signal: Arc::new(AtomicU32::new(0_u32))
        }
    }
}

impl Clone for QueryResults {
    fn clone(&self) -> Self {
        let signal_value = self.signal.load(Ordering::Relaxed);
        QueryResults {
            tr: self.tr.clone(),
            deadline: self.deadline.clone(),
            series: self.series.clone(),
            sr: self.sr.clone(),
            signal: Arc::new(AtomicU32::new(signal_value))
        }
    }
}

impl QueryResults {
    pub(crate) fn new() -> Self {
        QueryResults::default()
    }

    /// Len returns the number of results in rss.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    fn deadline_exceeded(&self) -> bool {
        return self.deadline.exceeded();
    }

    pub fn should_stop(&mut self) -> bool {
        if self.signal.fetch_add(0, Ordering::Relaxed) != 0 {
            return true
        }
        if self.deadline_exceeded() {
            self.signal.store(2, Ordering::SeqCst);
            return true;
        }
        return false;
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
    where F: Fn(Arc<&mut C>, &mut QueryResult, u64) -> RuntimeResult<()> + Send + Sync
    {
        let mut id = 0;
        for ts in self.series.iter_mut() {
            ts.worker_id = id;
            id += 1;
        }

        let must_stop = AtomicBool::new(false);
        let deadline = &self.deadline;
        let sharable_ctx = Arc::new(ctx);

        self.series
            .par_iter_mut()
            .filter(|rs| rs.timestamps.len() > 0)
            .try_for_each(|mut rs| {

            if must_stop.load(Ordering::Relaxed) {
                return Err(RuntimeError::TaskCancelledError("Search".to_string()));
            }

            if deadline.exceeded() {
                must_stop.store(true, Ordering::Relaxed);
                return Err(RuntimeError::deadline_exceeded("todo!!"));
            }

            let worker_id = rs.worker_id;
            f(Arc::clone(&sharable_ctx), &mut rs, worker_id)
        })

    }

    pub fn cancel(&mut self) {
        self.signal.store(1, Ordering::Relaxed);
    }
}

pub fn remove_empty_values_and_timeseries(tss: &mut Vec<QueryResult>) {
    tss.retain_mut(|ts| {

        for i in (ts.timestamps.len() .. 0).rev() {
            let v = ts.values[i];
            if v.is_nan() {
                ts.values.remove(i);
                ts.timestamps.remove(i);
            }
        }

        ts.values.len() > 0
    });
}


pub(crate) fn get_pooled_result<'a>() -> LinearReusable<'a, QueryResult> {
    get_result_pool().pull()
}

fn get_result_pool() -> &'static LinearObjectPool<QueryResult> {
    static INSTANCE: OnceCell<LinearObjectPool<QueryResult>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        LinearObjectPool::<QueryResult>::new(
            ||  {
                let mut res = QueryResult::new();
                res.last_reset_time = Timestamp::now();
                res
            },
            |v| {
                let current_time = Timestamp::now();
                let len = v.values.len();
                let cap = v.values.capacity();
                v.reset();
                if cap > 1024*1024 && 4 * len < cap && current_time - v.last_reset_time > 150 {
                    // Reset r.rs in order to preserve memory usage after processing big time series with
                    // millions of rows.
                    v.shrink(1024);
                }
            })
    })
}