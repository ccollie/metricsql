use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

static GLOBAL_DEDUP_INTERVAL: Lazy<AtomicI64> = Lazy::new(|| AtomicI64::new(0));

/// set_dedup_interval sets the deduplication interval, which is applied to raw samples during data
/// ingestion and querying.
///
/// De-duplication is disabled if dedup_interval is 0.
///
/// This function must be called before initializing the storage.
pub fn set_dedup_interval(dedup_interval: Duration) {
    GLOBAL_DEDUP_INTERVAL.swap(dedup_interval.as_millis() as i64, Ordering::Relaxed);
}

/// get_dedup_interval returns the dedup interval in milliseconds, which has been set via set_dedup_interval.
pub fn get_dedup_interval() -> i64 {
    return GLOBAL_DEDUP_INTERVAL.load(Ordering::Relaxed);
}

pub fn is_dedup_enabled() -> bool {
    return get_dedup_interval() > 0;
}

/// removes samples from src* if they are closer to each other than dedup_interval in milliseconds.
pub fn deduplicate_samples(src_timestamps: &mut Vec<i64>, src_values: &mut Vec<f64>, dedup_interval: i64) {
    if !needs_dedup(src_timestamps, dedup_interval)  {
        // Fast path - nothing to deduplicate
        return;
    }
    let mut ts_next = src_timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    let mut j: usize = 0;
    let mut count = 0;

    for i in 1 .. src_timestamps.len() {
        let ts = src_timestamps[i];
        if ts <= ts_next {
            continue;
        }

        src_timestamps[j] = ts;
        src_values[j] = src_values[i];
        j += 1;
        count += 1;

        ts_next = ts_next + dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }

    src_timestamps[count - 1] = src_timestamps[src_timestamps.len() - 1];
    src_values[count - 1] = src_values[src_values.len() - 1];
    src_timestamps.truncate(count);
    src_values.truncate(count);
}

pub fn deduplicate_samples_during_merge(src_timestamps: &mut Vec<i64>, src_values: &mut Vec<i64>, dedup_interval: i64) {
    if !needs_dedup(src_timestamps, dedup_interval)  {
        // Fast path - nothing to deduplicate
        return;
    }
    let mut ts_next = src_timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);

    let mut j: usize = 1;
    let mut count: usize = 0;

    for i in 1 .. src_timestamps.len() {
        let ts = src_timestamps[i];
        if ts <= ts_next {
            continue;
        }

        src_timestamps[j] = ts;
        src_values[j] = src_values[i];
        j += 1;
        count += 1;

        ts_next += dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    src_timestamps[count - 1] = src_timestamps[src_timestamps.len() - 1];
    src_values[count - 1] = src_values[src_values.len() - 1];
    src_timestamps.truncate(count);
    src_values.truncate(count);
}

fn needs_dedup(timestamps: &[i64], dedup_interval: i64) -> bool {
    if timestamps.len() < 2 || dedup_interval <= 0 {
        return false;
    }
    let mut ts_next = timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    for i in 1 .. timestamps.len() {
        let ts = timestamps[i];
        if ts <= ts_next {
            return true;
        }
        ts_next += dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    return false;
}