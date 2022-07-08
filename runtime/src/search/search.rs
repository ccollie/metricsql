use lockfree_object_pool::LinearObjectPool;
use once_cell::sync::OnceCell;
use std::fmt;
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use metricsql::ast::LabelFilter;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::search::Deadline;
use crate::{MetricName};
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
    // The time range for searching time series
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
        for tfs in self.tag_filterss {
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
    rows_processed: usize,
    /// Internal only
    worker_id: u64
}

impl QueryResult {
    pub fn new() -> Self {
        QueryResult {
            metric_name: MetricName::default(),
            values: vec![],
            timestamps: vec![],
            rows_processed: 0,
            worker_id: 0
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        QueryResult {
            metric_name: MetricName::default(),
            values: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
            rows_processed: 0,
            worker_id: 0
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
#[derive(Debug, Clone)]
pub struct QueryResults {
    pub tr: TimeRange,
    pub deadline: Deadline,
    pub series: Vec<QueryResult>,
    pub sr: Search,
    pub must_stop: Arc<AtomicBool>,
}

pub(crate) type SearchParallelFn = fn(rs: &mut QueryResult, worker_id: u64) -> RuntimeResult<()>;

impl Default for QueryResults {
    fn default() -> Self {
        Self {
            tr: TimeRange::default(),
            deadline: Deadline::default(),
            series: vec![],
            sr: Search::default(),
            must_stop: Arc::from(AtomicBool::new(false))
        }
    }
}

impl QueryResults {
    pub(crate) fn new() -> Self {
        QueryResults::default()
    }

    /// Len returns the number of results in rss.
    pub fn len(self) -> usize {
        self.series.len()
    }

    fn deadline_exceeded(self) -> bool {
        return self.deadline.exceeded();
    }

    fn should_stop(&mut self) -> bool {
        if self.must_stop.load(Ordering::Relaxed) {
            return true
        }
        if self.deadline_exceeded() {
            self.must_stop.store(true, Ordering::SeqCst);
            return true;
        }
        return false;
    }

    /// run_parallel runs f in parallel for all the results from rss.
    ///
    /// f shouldn't hold references to rs after returning.
    /// Data processing is immediately stopped if f returns an error.
    ///
    /// rss becomes unusable after the call to run_parallel.
    pub(crate) fn run_parallel<F>(&mut self, f: F) -> RuntimeResult<()>
    where F: Fn(&mut QueryResult,u64) -> RuntimeResult<()> + Sync
    {

        let mut id = 0;
        for ts in self.series.iter_mut() {
            ts.worker_id = id;
            id += 1;
        }

        // todo: fetch all ts from storage

        let must_stop = self.must_stop.clone();
        let deadline = &self.deadline;

        let mut err: Option<RuntimeError> = None;

        self.series.into_par_iter().try_for_each(|mut rs| {
            if must_stop.load(Ordering::Relaxed) {
                return None;
            }
            if deadline.exceeded() {
                must_stop.store(true, Ordering::Relaxed);
                return None;
            }

            if rs.timestamps.len() > 0 {
                match f(&mut rs, rs.worker_id) {
                    Err(e) => {
                        err = Some(e);
                        None
                    },
                    Ok(_) => Some(())
                }
            } else {
                Some(())
            }
        });

        if let Some(e) = err {
            Err(e)
        } else {
            Ok(())
        }
    }

    pub fn cancel(&mut self) {
        self.must_stop.store(true, Ordering::Relaxed);
    }
}


pub(crate) fn get_pooled_result() -> &'static mut QueryResult{
    let entry = get_result_pool().pull();
    &mut entry.rs
}

#[derive(Debug)]
struct ResultPoolEntry {
    rs: QueryResult,
    last_reset_time: Timestamp
}

impl ResultPoolEntry {
    fn new() -> Self {
        ResultPoolEntry {
            rs: QueryResult::new(),
            last_reset_time: Timestamp::now()
        }
    }

    fn reset(&mut self) {
        let current_time = Timestamp::now();
        let values = &self.rs.values;
        let cap = values.capacity();
        self.rs.reset();
        if cap > 1024*1024 && 4*values.len() < cap && current_time - self.last_reset_time > 100 {
            // Reset r.rs in order to preserve memory usage after processing big time series with
            // millions of rows.
            self.rs.shrink(1024);
        }
    }
}


fn get_result_pool() -> &'static LinearObjectPool<ResultPoolEntry> {
    static INSTANCE: OnceCell<LinearObjectPool<ResultPoolEntry>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        LinearObjectPool::<ResultPoolEntry>::new(
            ||  ResultPoolEntry::new(),
            |v| { v.reset() })
    })
}