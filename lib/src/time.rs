use chrono::{DateTime, NaiveDateTime, Utc};
use std::time::Instant;

pub type Time = Instant;

pub fn now() -> Instant {
    Instant::now()
}

/// Converts a nanosecond UTC timestamp into a DateTime structure
/// (which can have year, month, etc. extracted)
///
/// This is roughly equivelnt to ConvertTime
/// from <https://github.com/influxdata/flux/blob/1e9bfd49f21c0e679b42acf6fc515ce05c6dec2b/values/time.go#L35-L37>
pub fn timestamp_to_datetime(ts: i64) -> DateTime<Utc> {
    let secs = ts / 1_000_000_000;
    let nsec = ts % 1_000_000_000;
    // Note that nsec as u32 is safe here because modulo on a negative ts value
    // still produces a positive remainder.
    let datetime = NaiveDateTime::from_timestamp(secs, nsec as u32);
    DateTime::from_utc(datetime, Utc)
}
