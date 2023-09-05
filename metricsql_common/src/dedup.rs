/// removes samples from src* if they are closer to each other than dedup_interval in milliseconds.
pub fn deduplicate_samples(
    src_timestamps: &mut Vec<i64>,
    src_values: &mut Vec<f64>,
    dedup_interval: i64,
) {
    if !needs_dedup(src_timestamps, dedup_interval) {
        // Fast path - nothing to deduplicate
        return;
    }

    let mut ts_next = src_timestamps[0] + dedup_interval - 1;
    ts_next = ts_next - (ts_next % dedup_interval);
    let mut j: usize = 0;
    let mut count = 0;

    // todo: eliminate bounds checks
    for i in 1..src_timestamps.len() {
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
    for ts in &timestamps[1..] {
        let ts = *ts;
        if ts <= ts_next {
            return true;
        }
        ts_next += dedup_interval;
        if ts_next < ts {
            ts_next = ts + dedup_interval - 1;
            ts_next -= ts_next % dedup_interval
        }
    }
    false
}
