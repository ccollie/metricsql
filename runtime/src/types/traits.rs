use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::prelude::*;

// Unix timestamp in milliseconds.
// TODO: make this a newType
pub type Timestamp = i64;

pub trait TimestampTrait {
    fn from(v: i64) -> Self;
    fn from_secs(v: i64) -> Self;
    fn now() -> Self;
    fn from_systemtime(time: SystemTime) -> Self;
    fn add(&self, d: Duration) -> Self;
    fn sub(&self, d: Duration) -> Self;
    fn round_up_to_secs(&self) -> Self;
    fn to_string_millis(&self) -> String;
    fn to_rfc3339(&self) -> String;
}

impl TimestampTrait for Timestamp {
    fn from(v: i64) -> Self {
        v
    }

    fn from_secs(v: i64) -> Self {
        v * 1000
    }

    fn now() -> Self {
        Timestamp::from_systemtime(SystemTime::now())
    }

    fn from_systemtime(time: SystemTime) -> Timestamp {
        match time.duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let val =
                    duration.as_secs() * 1000 + u64::from(duration.subsec_nanos()) / 1_000_000;
                val as i64
            }
            Err(e) => panic!(
                "SystemTime before UNIX EPOCH! Difference: {:?}",
                e.duration()
            ),
        }
    }

    #[inline]
    fn add(&self, d: Duration) -> Self {
        // TODO: check for i64 overflow
        *self + 1000 * d.as_secs() as i64 + d.subsec_millis() as i64
    }

    #[inline]
    fn sub(&self, d: Duration) -> Self {
        // TODO: check for i64 overflow
        *self - 1000 * d.as_secs() as i64 - d.subsec_millis() as i64
    }

    #[inline]
    fn round_up_to_secs(&self) -> Self {
        (((*self - 1) as f64 / 1000.0) as i64 + 1) * 1000
    }

    fn to_string_millis(&self) -> String {
        match NaiveDateTime::from_timestamp_millis(self / 1000) {
            Some(ts) => ts.format("%Y-%m-%dT%H:%M:%S%.3f").to_string(),
            None => "".to_string(),
        }
    }

    fn to_rfc3339(&self) -> String {
        let naive_date_time = NaiveDateTime::from_timestamp_millis(*self).unwrap();
        let date_time = DateTime::<Utc>::from_utc(naive_date_time, Utc);
        date_time.to_rfc3339()
    }
}