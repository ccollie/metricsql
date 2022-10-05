use chrono::{DateTime, NaiveDateTime, Utc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub type Time = Instant;

pub fn now() -> Instant {
    Instant::now()
}

/// Converts a nanosecond UTC timestamp into a DateTime structure
/// (which can have year, month, etc. extracted)
///
/// This is roughly equivalent to ConvertTime
/// from <https://github.com/influxdata/flux/blob/1e9bfd49f21c0e679b42acf6fc515ce05c6dec2b/values/time.go#L35-L37>
pub fn timestamp_to_datetime(ts: i64) -> DateTime<Utc> {
    let secs = ts / 1_000_000_000;
    let nsec = ts % 1_000_000_000;
    // Note that nsec as u32 is safe here because modulo on a negative ts value
    // still produces a positive remainder.
    let datetime = NaiveDateTime::from_timestamp(secs, nsec as u32);
    DateTime::from_utc(datetime, Utc)
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

pub fn round_to_seconds(ms: i64) -> i64 {
    ms - &ms % 1000
}
