use std::collections::HashMap;
use std::sync::RwLock;
use chrono::{Utc};

use crate::eval::EvalConfig;
use crate::traits::{Timestamp, TimestampTrait};

struct Inner {
    id: u64,
    data: HashMap<u64, ActiveQueryEntry>,
}

pub struct ActiveQueries {
    inner: RwLock<Inner>,
}

#[derive(Clone, Debug)]
pub struct ActiveQueryEntry {
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub qid: u64,
    pub quoted_remote_addr: String,
    pub q: String,
    pub start_time: Timestamp,
}

impl ActiveQueries {
    pub(crate) fn new() -> Self {
        let id = Utc::now().timestamp_nanos() as u64;
        let inner = Inner {
            id,
            data: HashMap::new(),
        };
        ActiveQueries {
            inner: RwLock::new(inner)
        }
    }

    pub(crate) fn add(&mut self, ec: &EvalConfig, q: &str) -> u64 {
        self.add_ex(ec, q, Timestamp::now())
    }

    pub(crate) fn add_ex(&mut self, ec: &EvalConfig, q: &str, start_time: Timestamp) -> u64 {
        let inner = self.inner.write().unwrap();
        inner.id = inner.id + 1;

        let aqe = ActiveQueryEntry {
            start: ec.start,
            end: ec.end,
            step: ec.step,
            qid: inner.id,
            quoted_remote_addr: ec.quoted_remote_addr.unwrap_or("".to_string()),
            q: q.to_string(),
            start_time,
        };

        inner.data.insert(aqe.qid, aqe);

        return aqe.qid.clone();
    }

    pub(crate) fn remove(&mut self, qid: u64) {
        let inner = self.inner.write().unwrap();
        inner.data.remove(&qid);
    }

    pub fn get_all(&self) -> Vec<&ActiveQueryEntry> {
        let inner = self.inner.read().unwrap();
        let mut entries = inner.data.values()
            .collect::<Vec<&ActiveQueryEntry>>();

        entries.sort_by(|a, b| a.start_time.cmp(&b.start_time));
        entries
    }
}
