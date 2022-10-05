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
    pub start: Timestamp,
    pub end: Timestamp,
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

    pub(crate) fn register(&self, ec: &EvalConfig, q: &str) -> u64 {
        self.register_with_start(ec, q, Timestamp::now())
    }

    pub(crate) fn register_with_start(&self, ec: &EvalConfig, q: &str, start_time: Timestamp) -> u64 {
        let mut inner = self.inner.write().unwrap();
        let qid = inner.id + 1;
        inner.id = qid;

        let quoted_remote_addr = if ec.quoted_remote_addr.is_some() {
            ec.quoted_remote_addr.as_ref().unwrap().clone()
        } else {
            "".to_string()
        };

        let aqe = ActiveQueryEntry {
            start: ec.start,
            end: ec.end,
            step: ec.step,
            qid,
            quoted_remote_addr,
            q: q.to_string(),
            start_time,
        };

        inner.data.insert(qid, aqe);

        qid
    }

    pub(crate) fn remove(&self, qid: u64) {
        let mut inner = self.inner.write().unwrap();
        inner.data.remove(&qid);
    }

    pub fn get_all(&self) -> Vec<ActiveQueryEntry> {
        let inner = self.inner.read().unwrap();
        let mut entries = inner.data.values()
            .map(|x| x.clone())
            .collect::<Vec<_>>();

        entries.sort_by(|a, b| a.start_time.cmp(&b.start_time));

        entries
    }
}
