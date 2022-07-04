// Rust port of https://github.com/valyala/histogram
// Package histogram provides building blocks for fast histograms.
const infNeg: f64 = f64::NEG_INFINITY;
const infPos: f64 = f64::INFINITY;
const nan: f64 = f64::NAN;

const MAX_SAMPLES: usize = 1000;

const E10MIN: i32 = -9;
const E10MAX: i32 = 18;
const BUCKETS_PER_DECIMAL: u8 = 18;
const DECIMAL_BUCKETS_COUNT: i32 = (E10MAX - E10MIN);
const BUCKETS_COUNT: usize = (DECIMAL_BUCKETS_COUNT * BUCKETS_PER_DECIMAL) as usize;

const BUCKET_MULTIPLIER: f64 = (10 as i32).pow((1 / BUCKETS_PER_DECIMAL) as u32) as f64;

const LOWER_BUCKET_RANGE: String = format!("0...{:.3}", E10MIN.pow10());
const UPPER_BUCKET_RANGE: String = format!("{:.3}...+Inf", E10MAX.pow10());

const BUCKET_RANGES: Vec<String> = Vec::with_capacity(BUCKETS_COUNT);
// bucketRangesOnce sync.Once

// Histogram is a histogram for non-negative values with automatically created buckets.
//
// See https://medium.com/@valyala/improving-histogram-usability-for-prometheus-and-grafana-bc7e5df0e350
//
// Each bucket contains a counter for values in the given range.
// Each non-empty bucket is exposed via the following metric:
//
//     <metric_name>_bucket{<optional_tags>,vmrange="<start>...<end>"} <counter>
//
// Where:
//
//     - <metric_name> is the metric name passed to NewHistogram
//     - <optional_tags> is optional tags for the <metric_name>, which are passed to NewHistogram
//     - <start> and <end> - start and end values for the given bucket
//     - <counter> - the number of hits to the given bucket during Update* calls
//
// Histogram buckets can be converted to Prometheus-like buckets with `le` labels
// with `prometheus_buckets(<metric_name>_bucket)` function from PromQL extensions in VictoriaMetrics.
// (see https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL ):
//
//     prometheus_buckets(request_duration_bucket)
//
// Time series produced by the Histogram have better compression ratio comparing to
// Prometheus histogram buckets with `le` labels, since they don't include counters
// for all the previous buckets.
//
// Zero histogram is usable.
#[derive(Clone)]
pub struct Histogram {
    max: f64,
    min: f64,
    count: u64,
    lower: u64,
    upper: u64,
    sum: f64,
    decimal_buckets: Vec<Option<Vec<u64>>>
}

impl Histogram {
    // returns new fast histogram.
    pub fn new() -> Self {
        Histogram {
            max: f64::NEG_INFINITY,
            min: f64::INFINITY,
            count: 0,
            lower: 0,
            upper: 0,
            sum: 0.0,
            decimal_buckets: Vec::with_capacity(DECIMAL_BUCKETS_COUNT)
        }
        // reset
    }

    pub fn with_capacity(n: usize) -> Self {
        Histogram {
            max: f64::NEG_INFINITY,
            min: f64::INFINITY,
            count: 0,
            lower: 0,
            upper: 0,
            sum: 0.0,
            decimal_buckets: Vec::with_capacity(DECIMAL_BUCKETS_COUNT)
        }
        // reset
    }

    // Reset resets the histogram.
    pub fn reset(&mut self) {
        self.mu.Lock();
        for db in self.decimal_buckets {
            if db.is_none() {
                continue
            }
            db.unwrap().clear();
        }
        self.max = f64::NEG_INFINITY;
        self.min = f64::INFINITY;
        self.count = 0;
        self.lower = 0;
        self.upper = 0;
        self.sum = 0.0;
        self.mu.Unlock()
    }

    pub fn sum(&self) -> f64 {

        let sum = self.sum;

        return sum;
    }

    // Update updates h with v.
    //
    // Negative values and NaNs are ignored.
    pub fn update(&mut self, v: f64) {
        if v.is_nan() || v < 0.0 {
            // Skip NaNs and negative values.
            return
        }
        self.count = self.count + 1;
        let bucket_idx = (v.log10() - E10MIN) * BUCKETS_PER_DECIMAL;
        self.mu.Lock();
        self.sum += v;
        if bucket_idx < 0 {
            self.lower = self.lower + 1;
        } else if bucket_idx >= BUCKETS_COUNT {
            self.upper = self.upper + 1;
        } else {
            let mut idx = bucket_idx as i32;
            if bucket_idx == idx && idx > 0 {
                // Edge case for 10^n values, which must go to the lower bucket
                // according to Prometheus logic for `le`-based histograms.
                idx = idx - 1;
            }
            let decimal_bucket_idx = idx / BUCKETS_PER_DECIMAL;
            let offset = idx % BUCKETS_PER_DECIMAL;
            let mut db = self.decimal_buckets.get(decimal_bucket_idx);
            if db.is_none() {
                let v: Vec<u64> =  Vec::with_capacity(BUCKETS_PER_DECIMAL as usize);
                db = Some(v);
                self.decimal_buckets[decimal_bucket_idx] = db;
            }
            let slot = db.unwrap();
            slot[offset] = slot[offset] + 1;
        }
        self.mu.Unlock()
    }

    // VisitNonZeroBuckets calls f for all buckets with non-zero counters.
    //
    // vmrange contains "<start>...<end>" string with bucket bounds. The lower bound
    // isn't included in the bucket, while the upper bound is included.
    // This is required to be compatible with Prometheus-style histogram buckets
    // with `le` (less or equal) labels.
    pub fn visit_non_zero_buckets(&self, f: fn(&str, u64)) {
        self.mu.Lock();
        if self.lower > 0 {
            f(&*LOWER_BUCKET_RANGE, self.lower)
        }
        for (decimalBucketIdx, db) in self.decimal_buckets.iter().enumerate() {
            if db.is_none() {
                continue
            }
            for (offset, count) in db.iter().enumerate() {
                if count > 0 {
                    let bucket_idx = decimalBucketIdx * BUCKETS_PER_DECIMAL + offset;
                    let vm_range = get_vmrange(bucket_idx);
                    f(&vm_range, count)
                }
            }
        }
        if self.upper > 0 {
            f(UPPER_BUCKET_RANGE, self.upper)
        }
        self.mu.Unlock()
    }

    // quantile_sorted calculates the given quantile over a sorted list of values.
    //
    // It is expected that values won't contain NaN items.
    // The implementation mimics Prometheus implementation for compatibility's sake.
    pub fn quantile_sorted(phi: f64, values: &[f64]) -> f64 {
        if values.len() == 0 || phi.is_nan() {
            return f64::NAN;
        }
        if phi < 0.0 {
            return f64::NEG_INFINITY;
        }
        if phi > 1.0 {
            return f64::INFINITY;
        }
        let n = values.len();
        let rank = phi * (n - 1) as f64;

        let rank_floor = rank.floor();
        let lower_index = rank_floor.clamp(0.0, rank_floor);
        let upper_index = std::cmp::min(values.len() - 1, lower_index + 1);

        let weight = rank - rank_floor;
        return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
    }

    pub fn marshal_to(&self, prefix: &str, mut dst: &String) {
        let mut count_total = 0;

        self.visit_non_zero_buckets(|vmrange: &str, count: u64| {
            let tag = format!("vmrange={}", vmrange);
            let metric_name = add_tag(prefix, tag);
            let (name, labels) = split_metric_name(&metric_name);
            dst.push_str( format!("{}_bucket${} {}\n", name, labels, count).as_str() );
            count_total += count;
        });
        if count_total == 0 {
            return;
        }
        let (name, labels) = split_metric_name(prefix);
        let sum = self.sum;
        if sum.floor() == sum {
            let msg = format!("{}_sum{} {}\n", name, labels, sum.floor()).as_str();
            dst.push_str(msg );
        } else {
            let msg = format!("{}_sum{} {}\n", name, labels, sum).as_str();
            dst.push_str(msg );
        }
        let s = format!("{}_count{} {}\n", name, labels, count_total).as_str();
        dst.push_str(s);
    }
}


fn init_bucket_ranges() {
    let mut v = 10.powi(E10MIN as i32);
    let mut start = format_float(v);
    for i in 0 ..BUCKETS_COUNT {
        v *= BUCKET_MULTIPLIER;
        let end = format_float(v);
        BUCKET_RANGES[i] = format!("{}...{}", start, end);
        start = end;
    }
}

#[inline]
fn format_float(v: f64) -> String {
    format!("{:.3}", v)
}

fn get_vmrange(bucket_idx: usize) -> String {
    if !BUCKET_RANGES.len() {
        init_bucket_ranges();
    }
    return BUCKET_RANGES[bucket_idx];
}

// todo: move to lib ?
fn split_metric_name(name: &str) -> (String, String) {
    let parts: Vec<&str> = name.rsplit('{').collect();
    if parts.len() > 1 {
        return (parts[0].to_string(), parts[1].to_string())
    }
    return (parts[0].to_string(), "".to_string());
}

fn add_tag(name: &str, tag: String) -> String {
    let mut need_braces = false;
    if name.len() == 0 {
        need_braces = true;
    } else {
        let last_ch = name.chars().last().unwrap();
        need_braces = last_ch == '}';
    }
    if need_braces {
        return format!("{}{{{}}}", name, tag)
    }
    let chars = name.chars();
    return format!("{},{}}}", chars[0..chars.len()-1], tag)
}