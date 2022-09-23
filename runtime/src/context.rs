use chrono::{DateTime, Utc};

use crate::active_queries::ActiveQueries;
use crate::cache::rollup_result_cache::RollupResultCache;
use crate::parser_cache::ParseCache;
use crate::query_stats::query_stats::QueryStatsTracker;
use crate::traits::{NullSeriesDataSource, SeriesDataSource};

pub struct Context {
    pub parse_cache: ParseCache,
    pub query_stats: QueryStatsTracker,
    pub active_queries: ActiveQueries,
    pub rollup_result_cache: RollupResultCache,
    pub series_data: Box<dyn SeriesDataSource>, // mutex
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the query on the given time_range_msecs, which has been started at start_time.
    ///
    /// register_query must be called when the query is finished.
    pub(crate) fn register_query(&mut self, query: &str, time_range_msecs: i64, start_time: DateTime<Utc>) {
        self.query_stats.register_query(query, time_range_msecs, start_time)
    }

}

impl Default for Context {
    fn default() -> Self {
        Self {
            parse_cache: ParseCache::default(),
            query_stats: QueryStatsTracker::default(),
            active_queries: ActiveQueries::new(),
            rollup_result_cache: RollupResultCache::default(),
            series_data: Box::new(NullSeriesDataSource{}),
        }
    }
}