use std::time::Duration;

/// humanize converts given number to a human-readable format
/// by adding metric prefixes https://en.wikipedia.org/wiki/Metric_prefix
pub fn humanize(v: f64) -> String {
    let mut v = v;
    if v == 0f64 || v.is_nan() || v.is_infinite() {
        return format!("{:.4}", v);
    }
    let mut prefix: &str;
    if v.abs() >= 1.0 {
        prefix = "";
        for p in ["k", "M", "G", "T", "P", "E", "Z", "Y"] {
            if v.abs() < 1000.0 {
                break;
            }
            prefix = p;
            v /= 1000.0;
        }
        return format!("{:.4}{prefix}", v);
    }
    prefix = "";
    for p in ["m", "u", "n", "p", "f", "a", "z", "y"] {
        if v.abs() >= 1.0 {
            break;
        }
        prefix = p;
        v *= 1000.0;
    }
    format!("{:.4}{prefix}", v)
}

pub fn humanize_duration_ms(v: i64) -> String {
    let mut v = v;
    if v == 0 {
        return format!("{}ms", v);
    }
    let mut prefix: &str;
    if v.abs() >= 1_000 {
        prefix = "";
        for p in ["s", "m", "h", "d", "y"] {
            if v.abs() < 1_000 {
                break;
            }
            prefix = p;
            v /= 1_000;
        }
        return format!("{}{}", v, prefix);
    }
    prefix = "";
    for p in ["ms", "us", "ns", "ps", "fs", "as", "zs", "ys"] {
        if v.abs() >= 1_000 {
            break;
        }
        prefix = p;
        v *= 1_000;
    }
    format!("{}{}", v, prefix)
}

pub fn humanize_duration(v: &Duration) -> String {
    humanize_duration_ms(v.as_millis() as i64)
}

pub fn humanize_bytes(size: f64) -> String {
    let mut suffix: &str = "";
    let mut size = size;
    for p in ["ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"] {
        if size.abs() < 1024f64 {
            break;
        }
        suffix = p;
        size /= 1024.0
    }
    format!("{:.4}{suffix}", size)
}

const TB: u64 = 1 << 40;
const GB: u64 = 1 << 30;
const MB: u64 = 1 << 20;
const KB: u64 = 1 << 10;

/// Present size in human-readable form
pub fn human_readable_size(size: usize) -> String {
    let size = size as u64;
    let (value, unit) = {
        if size >= 2 * TB {
            (size as f64 / TB as f64, "TB")
        } else if size >= 2 * GB {
            (size as f64 / GB as f64, "GB")
        } else if size >= 2 * MB {
            (size as f64 / MB as f64, "MB")
        } else if size >= 2 * KB {
            (size as f64 / KB as f64, "KB")
        } else {
            (size as f64, "B")
        }
    };
    format!("{value:.1} {unit}")
}
