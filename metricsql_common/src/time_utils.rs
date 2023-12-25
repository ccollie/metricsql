use std::time::{Instant};

/// the majority of the functionality in this file is Licensed to the Apache Software Foundation (ASF)
/// under the APACHE License 2.0
/// http://www.apache.org/licenses/LICENSE-2.0
/// https://docs.rs/arrow-array/29.0.0/src/arrow_array/lib.rs.html
use chrono::{Utc};

/// Number of seconds in a day
pub const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
pub const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
pub const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
pub const NANOSECONDS: i64 = 1_000_000_000;

pub type Time = Instant;

pub fn now() -> Instant {
    Instant::now()
}

pub fn round_to_seconds(ms: i64) -> i64 {
    ms - &ms % 1000
}

/// Returns the time duration since UNIX_EPOCH in milliseconds.
pub fn current_time_millis() -> i64 {
    Utc::now().timestamp_millis()
}


#[cfg(test)]
mod tests {
}
