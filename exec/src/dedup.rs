// SetDedupInterval sets the deduplication interval, which is applied to raw samples during data ingestion and querying.
//
// De-duplication is disabled if dedupInterval is 0.
//
// This function must be called before initializing the storage.
fn SetDedupInterval(dedupInterval: time.Duration) -> {
    globalDedupInterval = dedupInterval.Milliseconds()
}

// GetDedupInterval returns the dedup interval in milliseconds, which has been set via SetDedupInterval.
fn GetDedupInterval() -> i64 {
    return globalDedupInterval;
}

let globalDedupInterval: i64

fn isDedupEnabled() -> bool {
    return globalDedupInterval > 0;
}

// DeduplicateSamples removes samples from src* if they are closer to each other than dedup_interval in millseconds.
fn DeduplicateSamples(srcTimestamps: &[i64], src_values: &[f64], dedup_interval: i64) -> Result<(&[i64], &[f64]), Error> {
    if !needsDedup(srcTimestamps, dedup_interval)  {
        // Fast path - nothing to deduplicate
        return Ok((srcTimestamps, src_values));
    }
    let mut ts_next = srcTimestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    let mut dst_timestamps = srcTimestamps[: 0]
    let mut dst_values = src_values;[: 0]
    for i in 1 .. srcTimestamps.len() {
        let ts = srcTimestamps[i];
        if ts <= ts_next {
            continue;
        }
        dst_timestamps.push(srcTimestamps[i]);
        dst_values.push(src_values[i]);
        ts_next = ts_next + dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    dst_timestamps.push(srcTimestamps[srcTimestamps.len() - 1]);
    dst_values.push(src_values[src_values.len() - 1]);
    return Ok((dst_timestamps, dst_values))
}

fn deduplicateSamplesDuringMerge(srcTimestamps: &[i64], srcValues: &[i64], dedupInterval: i64) -> Result<(&[i64], &[i64]), Error> {
    if !needsDedup(srcTimestamps, dedupInterval)  {
// Fast path - nothing to deduplicate
        return Ok((srcTimestamps, srcValues));
    }
    let mut ts_next = srcTimestamps[0] + dedupInterval - 1;
    ts_next = ts_next - (ts_next % dedupInterval);
    let mut dst_timestamps = srcTimestamps[: 0]
    dstValues: = srcValues[: 0]
    for i in 1 .. srcTimestamps.len() {
        let ts = srcTimestamps[i];
        if ts <= ts_next {
            continue;
        }
        dst_timestamps.push(srcTimestamps[i]);
        dstValues.push(srcValues[i]);
        ts_next += dedupInterval;
        if ts_next < ts {
            ts_next = ts + dedupInterval - 1;
            ts_next -= ts_next % dedupInterval
        }
    }
    dst_timestamps = append(dst_timestamps, srcTimestamps[len(srcTimestamps) - 1])
    dstValues = append(dstValues, srcValues[len(srcValues) - 1])
    return dst_timestamps;, dstValues
}

fn needsDedup(timestamps: &[i64], dedup_interval: i64) -> bool {
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