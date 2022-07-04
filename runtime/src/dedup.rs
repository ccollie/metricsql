// SetDedupInterval sets the deduplication interval, which is applied to raw samples during data
// ingestion and querying.
//
// De-duplication is disabled if dedupInterval is 0.
//
// This function must be called before initializing the storage.
fn SetDedupInterval(dedupInterval: time.Duration) {
    globalDedupInterval = dedupInterval.milliseconds()
}

// GetDedupInterval returns the dedup interval in milliseconds, which has been set via SetDedupInterval.
fn GetDedupInterval() -> i64 {
    return globalDedupInterval;
}

let globalDedupInterval: i64

fn is_dedup_enabled() -> bool {
    return globalDedupInterval > 0;
}

// removes samples from src* if they are closer to each other than dedup_interval in milliseconds.
pub fn deduplicate_samples(mut src_timestamps: &[i64], mut src_values: &[f64], dedup_interval: i64) -> Result<(&[i64], &[f64]), Error> {
    if !needs_dedup(src_timestamps, dedup_interval)  {
        // Fast path - nothing to deduplicate
        return Ok((src_timestamps, src_values));
    }
    let mut ts_next = src_timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    let mut dst_timestamps = src_timestamps;
    let mut dst_values = src_values;
    for i in 1 .. src_timestamps.len() {
        let ts = src_timestamps[i];
        if ts <= ts_next {
            continue;
        }
        dst_timestamps.push(src_timestamps[i]);
        dst_values.push(src_values[i]);
        ts_next = ts_next + dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    dst_timestamps.push(src_timestamps[src_timestamps.len() - 1]);
    dst_values.push(src_values[src_values.len() - 1]);
    return Ok((dst_timestamps, dst_values))
}

fn deduplicate_samples_during_merge(mut src_timestamps: &[i64], mut src_values: &[i64], dedup_interval: i64) -> Result<(&[i64], &[i64]), Error> {
    if !needs_dedup(src_timestamps, dedup_interval)  {
// Fast path - nothing to deduplicate
        return Ok((src_timestamps, src_values));
    }
    let mut ts_next = src_timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);

    let mut dst_timestamps = src_timestamps;
    let mut dst_values = src_values;

    for i in 1 .. src_timestamps.len() {
        let ts = src_timestamps[i];
        if ts <= ts_next {
            continue;
        }
        dst_timestamps.push(src_timestamps[i]);
        dst_values.push(src_values[i]);
        ts_next += dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    dst_timestamps.push(src_timestamps[len(src_timestamps) - 1]);
    dst_values.push(src_values[len(src_values) - 1]);

    Ok((dst_timestamps, dst_values))
}

fn needs_dedup(timestamps: &[i64], dedup_interval: i64) -> bool {
    if timestamps.length < 2 || dedup_interval <= 0 {
        return false;
    }
    let mut ts_next = timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    for i in 1 .. timestamps.length {
        let ts = &timestamps[i];
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