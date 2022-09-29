use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Add, Sub};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::Duration;
use chrono::prelude::Utc;
use chrono::prelude::DateTime;

const QUERY_STATS_DEFAULT_CAPACITY: usize = 250;

#[derive(Hash, Clone, Default, PartialEq, Eq)]
pub struct QueryStatKey {
    pub query: String,
    pub time_range_secs: i64,
}

#[derive(Hash, Clone)]
pub struct QueryStatRecord {
    pub key: QueryStatKey,
    pub register_time: DateTime<Utc>,
    pub duration: Duration,
}

impl QueryStatRecord {
    fn matches(&self, current_time: DateTime<Utc>, max_lifetime: Duration) -> bool {
        if self.key.query.len() == 0 {
            return false
        }
        let elapsed = current_time.sub(self.register_time);
        if elapsed.cmp( &max_lifetime) == Ordering::Greater {
            return false;
        }
        return true;
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
            count: 0
        }
    }
}

#[derive(Hash, Clone)]
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


struct Inner {
    data: Vec<QueryStatRecord>,
    next_idx: usize,
}

/// QueryStatsTracker holds statistics for queries
pub struct QueryStatsTracker {
    inner: Arc<Mutex<Inner>>,
    config: QueryStatsConfig
}

impl Default for QueryStatsTracker  {
    fn default() -> Self {
        Self::new(QueryStatsConfig::new(), QUERY_STATS_DEFAULT_CAPACITY)
    }
}

impl QueryStatsTracker {
    pub fn new(config: QueryStatsConfig, cap: usize) -> Self {
        let inner = Inner {
            data: Vec::with_capacity(cap),
            next_idx: 0
        };

        QueryStatsTracker {
            inner: Arc::new(Mutex::new(inner)),
            config: config.clone()
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.config.last_queries_count > 0
    }

    pub(crate) fn register_query(&mut self, query: &str, time_range_msecs: i64, start_time: DateTime<Utc>) {
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
                time_range_secs
            }
        };

        {
            let mut qst = self.inner.lock().unwrap();
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

    fn get_top_by_count(self, top_n: usize, max_lifetime: Duration) -> Vec<QueryStatByCount> {
        let current_time = Utc::now();

        let mut m: HashMap<&QueryStatKey, u64> = HashMap::new();
        let qst = self.inner.lock().unwrap();
        qst.data.iter().for_each(|r|  {
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
                count: count as u64,
            })
        }
        a.sort_by(|a, b| a.count.cmp(&b.count));
        if a.len() > top_n {
            a.resize(top_n, QueryStatByCount::default());
        }

        return a;
    }

    fn get_top_by_avg_duration(self, top_n: usize, max_lifetime: Duration) -> Vec<QueryStatByDuration> {
        let current_time = Utc::now();

        #[derive(Hash, Copy, Clone)]
        struct CountSum {
            count: usize,
            sum: Duration,
        }

        let mut m: HashMap<&QueryStatKey, CountSum> = HashMap::new();

        let inner = self.inner.lock().unwrap();

        inner.data.iter().for_each(|r| {
            if r.matches(current_time, max_lifetime) {
                let k = &r.key;
                let mut ks = m.entry(k)
                    .or_insert(CountSum{
                        count: 0,
                        sum: Duration::milliseconds(0)
                    });

                ks.count += 1;
                ks.sum.add(r.duration);
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
        return a;
    }

    fn get_top_by_sum_duration(&mut self, top_n: usize, max_lifetime: Duration) -> Vec<QueryStatByDuration> {
        let current_time = Utc::now();

        #[derive(Hash, Clone)]
        struct CountDuration {
            count: usize,
            sum: Duration,
        }

        let mut m: HashMap<&QueryStatKey, CountDuration> = HashMap::new();

        let qst = self.inner.lock().unwrap();
        qst.data.iter().for_each(|r| {
            if r.matches(current_time, max_lifetime) {
                let kd = m.entry(&r.key).or_insert(
                    CountDuration{
                        count: 0,
                        sum: Duration::milliseconds(0)
                    }
                );
                kd.count +=1;
                kd.sum.add(r.duration);
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
        return a;
    }
}

pub fn systemtime_to_timestamp(time: SystemTime) -> u64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs() * 1000 + u64::from(duration.subsec_nanos()) / 1_000_000,
        Err(e) => panic!(
            "SystemTime before UNIX EPOCH! Difference: {:?}",
            e.duration()
        ),
    }
}