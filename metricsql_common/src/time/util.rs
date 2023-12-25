// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Part of the functionality in this file is Licensed to the Apache Software Foundation (ASF)
/// under the APACHE License 2.0
/// http://www.apache.org/licenses/LICENSE-2.0
/// https://docs.rs/arrow-array/29.0.0/src/arrow_array/lib.rs.html
///
use std::str::FromStr;

use chrono::{Datelike, DateTime, LocalResult, NaiveDate, NaiveDateTime, Timelike, TimeZone, Utc, Weekday};
use chrono_tz::Tz;

/// Number of seconds in a day
pub const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
pub const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
pub const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
pub const NANOSECONDS: i64 = 1_000_000_000;

use super::timezone::get_timezone;

pub fn format_utc_datetime(utc: &NaiveDateTime, pattern: &str) -> String {
    match get_timezone(None) {
        super::Timezone::Offset(offset) => {
            offset.from_utc_datetime(utc).format(pattern).to_string()
        }
        super::Timezone::Named(tz) => tz.from_utc_datetime(utc).format(pattern).to_string(),
    }
}

pub fn local_datetime_to_utc(local: &NaiveDateTime) -> LocalResult<NaiveDateTime> {
    match get_timezone(None) {
        super::Timezone::Offset(offset) => offset.from_local_datetime(local).map(|x| x.naive_utc()),
        super::Timezone::Named(tz) => tz.from_local_datetime(local).map(|x| x.naive_utc()),
    }
}

/// converts a `i64` representing a `timestamp(ms)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ms_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, milli_sec) = split_second(v, MILLISECONDS);

    NaiveDateTime::from_timestamp_opt(
        // extract seconds from milliseconds
        sec,
        // discard extracted seconds and convert milliseconds to nanoseconds
        milli_sec * MICROSECONDS as u32,
    )
}

/// converts a `i64` representing a `timestamp(s)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_s_to_datetime(v: i64) -> Option<NaiveDateTime> {
    NaiveDateTime::from_timestamp_opt(v, 0)
}

/// converts a `i64` representing a `time64(s)` to [`NaiveDateTime`]
pub fn timestamp_secs_to_datetime(secs: i64) -> Option<NaiveDateTime> {
    NaiveDateTime::from_timestamp_opt(
        // extract seconds from seconds
        secs, 0_u32,
    )
}

/// converts a `i64` representing a `time64(s)` to [`DateTime<Utc>`]
pub fn timestamp_secs_to_utc_datetime(secs: i64) -> Option<DateTime<Utc>> {
    timestamp_secs_to_datetime(secs).map(|naive| Utc.from_utc_datetime(&naive))
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, micro_sec) = split_second(v, MICROSECONDS);

    NaiveDateTime::from_timestamp_opt(
        // extract seconds from microseconds
        sec,
        // discard extracted seconds and convert microseconds to nanoseconds
        micro_sec * MILLISECONDS as u32,
    )
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, nano_sec) = split_second(v, NANOSECONDS);

    NaiveDateTime::from_timestamp_opt(
        // extract seconds from nanoseconds
        sec, // discard extracted seconds
        nano_sec,
    )
}

/// Converts a nanosecond UTC timestamp into a DateTime structure
/// (which can have year, month, etc. extracted)
///
/// This is roughly equivalent to ConvertTime
/// from <https://github.com/influxdata/flux/blob/1e9bfd49f21c0e679b42acf6fc515ce05c6dec2b/values/time.go#L35-L37>
pub fn timestamp_to_datetime(ts: i64) -> Option<DateTime<Utc>> {
    timestamp_ns_to_datetime(ts).map(|naive| Utc.from_utc_datetime(&naive))
}

pub fn find_tz_from_env() -> Option<Tz> {
    // Windows does not support "TZ" env variable, which is used in the `Local` timezone under Unix.
    // However, we are used to set "TZ" env as the default timezone without actually providing a
    // timezone argument (especially in tests), and it's very convenient to do so, we decide to make
    // it work under Windows as well.
    std::env::var("TZ")
        .ok()
        .and_then(|tz| Tz::from_str(&tz).ok())
}

pub fn get_local_tz() -> Option<Tz> {
    find_tz_from_env().or_else(|| Tz::from_str("UTC").ok())
}

/// Returns the time duration since UNIX_EPOCH in milliseconds.
pub fn current_time_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Returns the current time in rfc3339 format.
pub fn current_time_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Returns the yesterday time in rfc3339 format.
pub fn yesterday_rfc3339() -> String {
    let now = chrono::Utc::now();
    let day_before = now - chrono::Duration::days(1);
    day_before.to_rfc3339()
}

/// Port of rust unstable features `int_roundings`.
pub(crate) fn div_ceil(this: i64, rhs: i64) -> i64 {
    let d = this / rhs;
    let r = this % rhs;
    if r > 0 && rhs > 0 {
        d + 1
    } else {
        d
    }
}

#[inline]
pub(crate) fn split_second(v: i64, base: i64) -> (i64, u32) {
    (v.div_euclid(base), v.rem_euclid(base) as u32)
}

pub fn is_leap_year(y: u32) -> bool {
    if y % 4 != 0 {
        return false;
    }
    if y % 100 != 0 {
        return true;
    }
    y % 400 == 0
}

const DAYS_IN_MONTH: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

pub fn days_in_month<Tz: TimeZone>(t: DateTime<Tz>) -> u8 {
    let m = t.month() as usize;
    let y = t.year() as u32;
    if m == 2 && is_leap_year(y) {
        return 29_u8;
    }
    DAYS_IN_MONTH[m - 1]
}

pub fn int_day_of_week<Tz: TimeZone>(t: DateTime<Tz>) -> u8 {
    match t.weekday() {
        Weekday::Sun => 0_u8,
        Weekday::Mon => 1_u8,
        Weekday::Tue => 2_u8,
        Weekday::Wed => 3_u8,
        Weekday::Thu => 4_u8,
        Weekday::Fri => 5_u8,
        Weekday::Sat => 6_u8,
    }
}


#[derive(Clone, Copy)]
pub enum DateTimePart {
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    DaysInMonth,
    Hour,
    Minute,
    Month,
    Second,
    Year,
}

impl DateTimePart {
    pub fn get_part_from_datetime<Tz: TimeZone>(self, datetime: DateTime<Tz>) -> Option<u32> {
        datetime_part(datetime, self)
    }

    pub fn get_part_from_naive_datetime(&self, datetime: NaiveDateTime) -> u32 {
        match self {
            DateTimePart::DayOfMonth => datetime.day(),
            DateTimePart::DayOfWeek => datetime.weekday().num_days_from_sunday(),
            DateTimePart::DayOfYear => datetime.ordinal(),
            DateTimePart::DaysInMonth => {
                let cur_month = datetime.month();
                let cur_year = datetime.year();
                let naive_date = if cur_month == 12 {
                    NaiveDate::from_ymd_opt(cur_year + 1, 1, 1)
                } else {
                    NaiveDate::from_ymd_opt(cur_year, cur_month + 1, 1)
                };
                naive_date
                    .unwrap()
                    .signed_duration_since(NaiveDate::from_ymd_opt(cur_year, cur_month, 1).unwrap())
                    .num_days() as u32
            }
            DateTimePart::Hour => datetime.hour(),
            DateTimePart::Minute => datetime.minute(),
            DateTimePart::Month => datetime.month(),
            DateTimePart::Second => datetime.second(),
            DateTimePart::Year => datetime.year() as u32,
        }
    }

    pub fn get_part_from_timestamp_ms(&self, timestamp: i64) -> Option<u32> {
        if let Some(datetime) = timestamp_ms_to_datetime(timestamp) {
            return Some(self.get_part_from_naive_datetime(datetime));
        }
        None
    }
}

pub fn datetime_part<Tz: TimeZone>(datetime: DateTime<Tz>, part: DateTimePart) -> Option<u32> {
    match part {
        DateTimePart::DayOfMonth => Some(datetime.day()),
        DateTimePart::DayOfWeek => Some(int_day_of_week(datetime) as u32),
        DateTimePart::DayOfYear => Some(datetime.ordinal()),
        DateTimePart::DaysInMonth => Some(days_in_month(datetime) as u32),
        DateTimePart::Hour => Some(datetime.hour()),
        DateTimePart::Minute => Some(datetime.minute()),
        DateTimePart::Month => Some(datetime.month()),
        DateTimePart::Second => Some(datetime.second()),
        DateTimePart::Year => {
            let year = datetime.year();
            // we don't handle negative years currently
            if year < 0 {
                None
            } else {
                Some(year as u32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{self, SystemTime};

    use chrono::{Datelike, TimeZone, Timelike};

    use super::*;

    #[test]
    fn test_current_time_millis() {
        let now = current_time_millis();

        let millis_from_std = SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let datetime_now = chrono::Utc.timestamp_millis_opt(now).unwrap();
        let datetime_std = chrono::Utc.timestamp_millis_opt(millis_from_std).unwrap();

        assert_eq!(datetime_std.year(), datetime_now.year());
        assert_eq!(datetime_std.month(), datetime_now.month());
        assert_eq!(datetime_std.day(), datetime_now.day());
        assert_eq!(datetime_std.hour(), datetime_now.hour());
        assert_eq!(datetime_std.minute(), datetime_now.minute());
    }

    #[test]
    fn test_div_ceil() {
        let v0 = 9223372036854676001;
        assert_eq!(9223372036854677, div_ceil(v0, 1000));
    }

    #[test]
    fn negative_input_timestamp_ns_to_datetime() {
        assert_eq!(
            timestamp_ns_to_datetime(-1),
            NaiveDateTime::from_timestamp_opt(-1, 999_999_999)
        );

        assert_eq!(
            timestamp_ns_to_datetime(-1_000_000_001),
            NaiveDateTime::from_timestamp_opt(-2, 999_999_999)
        );
    }

    #[test]
    fn negative_input_timestamp_us_to_datetime() {
        assert_eq!(
            timestamp_us_to_datetime(-1),
            NaiveDateTime::from_timestamp_opt(-1, 999_999_000)
        );

        assert_eq!(
            timestamp_us_to_datetime(-1_000_001),
            NaiveDateTime::from_timestamp_opt(-2, 999_999_000)
        );
    }

    #[test]
    fn negative_input_timestamp_ms_to_datetime() {
        assert_eq!(
            timestamp_ms_to_datetime(-1),
            NaiveDateTime::from_timestamp_opt(-1, 999_000_000)
        );

        assert_eq!(
            timestamp_ms_to_datetime(-1_001),
            NaiveDateTime::from_timestamp_opt(-2, 999_000_000)
        );
    }

    #[test]
    fn test_split_seconds() {
        let (sec, nano_sec) = split_second(100, NANOSECONDS);
        assert_eq!(sec, 0);
        assert_eq!(nano_sec, 100);

        let (sec, nano_sec) = split_second(123_000_000_456, NANOSECONDS);
        assert_eq!(sec, 123);
        assert_eq!(nano_sec, 456);

        let (sec, nano_sec) = split_second(-1, NANOSECONDS);
        assert_eq!(sec, -1);
        assert_eq!(nano_sec, 999_999_999);

        let (sec, nano_sec) = split_second(-123_000_000_001, NANOSECONDS);
        assert_eq!(sec, -124);
        assert_eq!(nano_sec, 999_999_999);
    }
}
