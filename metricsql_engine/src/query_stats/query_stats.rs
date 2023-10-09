use std::cmp::Ordering;
use std::ops::Sub;
use std::sync::RwLock;

use ahash::AHashMap;
use chrono::prelude::DateTime;
use chrono::prelude::Utc;
use chrono::Duration;

const QUERY_STATS_DEFAULT_CAPACITY: usize = 250;

#[derive(Hash, Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryStatKey {
    pub query: String,
    pub time_range_secs: i64,
}

#[derive(Hash, Clone, Debug)]
pub struct QueryStatRecord {
    pub key: QueryStatKey,
    pub register_time: DateTime<Utc>,
    pub duration: Duration,
}

impl QueryStatRecord {
    pub(crate) fn matches(&self, current_time: DateTime<Utc>, max_lifetime: Duration) -> bool {
        if self.key.query.is_empty() {
            return false;
        }
        let elapsed = current_time.sub(self.register_time);
        if elapsed.cmp(&max_lifetime) == Ordering::Greater {
            return false;
        }
        true
    }
}

#[derive(Hash, Clone, Default)]
pub struct QueryStatByCount {
    pub query: String,
    pub time_range_secs: i64,
    pub count: u64,
}

#[derive(Hash, Clone)]
pub struct QueryStatByDuration {
    pub query: String,
    pub time_range_secs: i64,
    pub duration: Duration,
    pub count: u64,
}

impl Default for QueryStatByDuration {
    fn default() -> Self {
        Self {
            query: "".to_string(),
            time_range_secs: 0,
            duration: Duration::seconds(0),
            count: 0,
        }
    }
}

#[derive(Hash, Clone, Debug)]
/// Configuration settings for QueryStatsTracker
pub struct QueryStatsConfig {
    /// Zero value disables query stats tracking
    pub last_queries_count: usize,
    /// The minimum duration for queries to track in query stats
    /// Queries with lower duration are ignored in query stats
    pub min_query_duration: Duration,
}

impl QueryStatsConfig {
    pub fn new() -> Self {
        Self {
            last_queries_count: 1000,
            min_query_duration: Duration::milliseconds(1),
        }
    }
}

impl Default for QueryStatsConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Inner {
    data: Vec<QueryStatRecord>, // use deque ???
    next_idx: usize,
}

/// QueryStatsTracker holds statistics for queries
#[derive(Debug)]
pub struct QueryStatsTracker {
    inner: RwLock<Inner>,
    config: QueryStatsConfig,
}

impl Default for QueryStatsTracker {
    fn default() -> Self {
        Self::new(QueryStatsConfig::new(), QUERY_STATS_DEFAULT_CAPACITY)
    }
}

impl QueryStatsTracker {
    pub fn new(config: QueryStatsConfig, cap: usize) -> Self {
        let inner = Inner {
            data: Vec::with_capacity(cap),
            next_idx: 0,
        };

        QueryStatsTracker {
            inner: RwLock::new(inner),
            config: config.clone(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.last_queries_count > 0
    }

    /// Registers the query on the given time_range_msecs, which has been started at start_time.
    ///
    /// register_query must be called when the query is finished.
    pub(crate) fn register_query(
        &self,
        query: &str,
        time_range_msecs: i64,
        start_time: DateTime<Utc>,
    ) {
        let register_time = Utc::now();
        let duration = register_time.sub(start_time);
        if duration.cmp(&self.config.min_query_duration) == Ordering::Less {
            return;
        }

        let time_range_secs = time_range_msecs / 1000;
        let r = QueryStatRecord {
            register_time,
            duration,
            key: QueryStatKey {
                query: query.to_string(),
                time_range_secs,
            },
        };

        {
            let mut qst = self.inner.write().unwrap();
            let mut idx = qst.next_idx;
            let len = qst.data.len();
            if idx >= len {
                qst.data.pop();
                idx = 0;
            }
            qst.data.push(r);
            qst.next_idx = idx + 1;
        }
    }

    pub fn get_top_by_count(&self, top_n: usize, max_lifetime: Duration) -> Vec<QueryStatByCount> {
        let current_time = Utc::now();

        let mut m: AHashMap<&QueryStatKey, u64> = AHashMap::new();
        let qst = self.inner.read().unwrap();
        qst.data.iter().for_each(|r: &QueryStatRecord| {
            if r.matches(current_time, max_lifetime) {
                let entry = m.entry(&r.key).or_insert(0);
                *entry += 1;
            }
        });

        let mut a: Vec<QueryStatByCount> = Vec::with_capacity(m.len());
        for (k, count) in m {
            a.push(QueryStatByCount {
                query: k.query.clone(),
                time_range_secs: k.time_range_secs,
                count,
            })
        }
        a.sort_by(|a, b| a.count.cmp(&b.count));
        if a.len() > top_n {
            a.resize(top_n, QueryStatByCount::default());
        }

        a
    }

    pub fn get_top_by_avg_duration(
        &self,
        top_n: usize,
        max_lifetime: Duration,
    ) -> Vec<QueryStatByDuration> {
        let current_time = Utc::now();

        #[derive(Hash, Copy, Clone)]
        struct CountSum {
            count: usize,
            sum: Duration,
        }

        let mut m: AHashMap<&QueryStatKey, CountSum> = AHashMap::new();

        let inner = self.inner.read().unwrap();

        inner.data.iter().for_each(|r: &QueryStatRecord| {
            if r.matches(current_time, max_lifetime) {
                let k = &r.key;
                let ks = m.entry(k).or_insert(CountSum {
                    count: 0,
                    sum: Duration::milliseconds(0),
                });

                ks.count += 1;
                ks.sum = ks.sum + r.duration;
            }
        });

        let mut a: Vec<QueryStatByDuration> = Vec::with_capacity(m.len());
        for (k, ks) in m.iter() {
            a.push(QueryStatByDuration {
                query: k.query.clone(),
                time_range_secs: k.time_range_secs,
                duration: Duration::milliseconds(ks.sum.num_milliseconds() / ks.count as i64),
                count: ks.count as u64,
            })
        }
        a.sort_by(|a, b| a.duration.cmp(&b.duration));

        if a.len() > top_n {
            a.resize(top_n, QueryStatByDuration::default());
        }
        a
    }

    pub fn get_top_by_sum_duration(
        &self,
        top_n: usize,
        max_lifetime: Duration,
    ) -> Vec<QueryStatByDuration> {
        let current_time = Utc::now();

        #[derive(Hash, Clone)]
        struct CountDuration {
            count: usize,
            sum: Duration,
        }

        let mut m: AHashMap<&QueryStatKey, CountDuration> = AHashMap::new();

        let qst = self.inner.read().unwrap();
        qst.data.iter().for_each(|r: &QueryStatRecord| {
            if r.matches(current_time, max_lifetime) {
                let kd = m.entry(&r.key).or_insert(CountDuration {
                    count: 0,
                    sum: Duration::milliseconds(0),
                });
                kd.count += 1;
                kd.sum = kd.sum + r.duration;
            }
        });

        let mut a: Vec<QueryStatByDuration> = Vec::with_capacity(m.len());
        for (k, kd) in m.iter() {
            a.push(QueryStatByDuration {
                query: k.query.clone(),
                time_range_secs: k.time_range_secs,
                duration: kd.sum,
                count: kd.count as u64,
            })
        }
        a.sort_by(|a, b| a.duration.cmp(&b.duration));
        if a.len() > top_n {
            a.resize(top_n, QueryStatByDuration::default());
        }
        a
    }
}
