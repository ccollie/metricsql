use lockfree_object_pool::{LinearObjectPool, LinearReusable};
use once_cell::sync::{Lazy, OnceCell};
use std::fmt;
use std::fmt::Display;

const E10MIN: i32 = -9;
const E10MAX: i32 = 18;
const BUCKETS_PER_DECIMAL: usize = 18;
const DECIMAL_BUCKETS_COUNT: i32 = E10MAX - E10MIN;
const BUCKETS_COUNT: usize = (DECIMAL_BUCKETS_COUNT * BUCKETS_PER_DECIMAL as i32) as usize;

const LOWER_MIN: f64 = 1e-9;
const UPPER_MAX: f64 = 1e18;
const BUCKET_MULTIPLIER: f64 = 10_i32.pow((1 / BUCKETS_PER_DECIMAL) as u32) as f64;

static LOWER_BUCKET_RANGE: Lazy<String> = Lazy::new(|| format!("0...{:.3}", LOWER_MIN));
static UPPER_BUCKET_RANGE: Lazy<String> = Lazy::new(|| format!("{:.3}...+Inf", UPPER_MAX));

pub trait HistogramBucketVisitor {
    fn visit(vm_range: &str, count: u64);
}

/// Histogram is a histogram for non-negative values with automatically created buckets.
///
/// See https://medium.com/@valyala/improving-histogram-usability-for-prometheus-and-grafana-bc7e5df0e350
///
/// Each bucket contains a counter for values in the given range.
/// Each non-empty bucket is exposed via the following metric:
///
///     <metric_name>_bucket{<optional_tags>,vmrange="<start>...<end>"} <counter>
///
/// Where:
///
///     - <metric_name> is the metric name passed to NewHistogram
///     - <optional_tags> is optional tags for the <metric_name>, which are passed to NewHistogram
///     - <start> and <end> - start and end values for the given bucket
///     - <counter> - the number of hits to the given bucket during Update* calls
///
/// Histogram buckets can be converted to Prometheus-like buckets with `le` labels
/// with `prometheus_buckets(<metric_name>_bucket)` function from PromQL extensions in VictoriaMetrics.
/// (see https://github.com/VictoriaMetrics/VictoriaMetrics/wiki/MetricsQL ):
///
///     prometheus_buckets(request_duration_bucket)
///
/// Time series produced by the Histogram have better compression ratio comparing to
/// Prometheus histogram buckets with `le` labels, since they don't include counters
/// for all the previous buckets.
///
/// Zero histogram is usable.
#[derive(Clone, Debug, Default)]
pub struct Histogram {
    count: u64,
    lower: u64,
    upper: u64,
    sum: f64,
    decimal_buckets: Vec<Vec<u64>>,
    values: Vec<f64>
}

impl Histogram {
    // returns new fast histogram.
    pub fn new() -> Self {
        Self::with_capacity(DECIMAL_BUCKETS_COUNT as usize)
    }

    pub fn with_capacity(n: usize) -> Self {
        let mut buckets: Vec<Vec<u64>> = Vec::with_capacity(n);
        buckets.resize_with(n, || vec![0]);
        Histogram {
            count: 0,
            lower: 0,
            upper: 0,
            sum: 0.0,
            decimal_buckets: buckets,
            values: Vec::with_capacity(16)
        }
        // reset
    }

    // Reset resets the histogram.
    pub fn reset(&mut self) {
        for bucket in self.decimal_buckets.iter_mut() {
            for v in bucket {
                *v = 0;
            }
        }
        self.count = 0;
        self.lower = 0;
        self.upper = 0;
        self.values.clear();
    }

    // Update updates h with v.
    //
    // Negative values and NaNs are ignored.
    pub fn update(&mut self, v: f64) {
        if v.is_nan() || v < 0.0 {
            // Skip NaNs and negative values.
            return;
        }
        self.count += 1;
        let bucket_idx = (v.log10() - E10MIN as f64) * BUCKETS_PER_DECIMAL as f64;
        self.sum += v;
        self.values.push(v);
        if bucket_idx < 0_f64 {
            self.lower += 1;
        } else if bucket_idx >= BUCKETS_COUNT as f64 {
            self.upper += 1;
        } else {
            let mut idx = bucket_idx.floor() as usize;
            if bucket_idx == idx as f64 {
                // Edge case for 10^n values, which must go to the lower bucket
                // according to Prometheus logic for `le`-based histograms.
                idx -= 1;
            }
            let decimal_bucket_idx = (idx / BUCKETS_PER_DECIMAL) as usize;
            let offset = (idx % BUCKETS_PER_DECIMAL) as usize;

            self.decimal_buckets[decimal_bucket_idx].reserve(BUCKETS_PER_DECIMAL);
            self.decimal_buckets[bucket_idx as usize][offset] += 1;
        }
    }

    pub fn marshal_to(&self, prefix: &str, dst: &mut String) {
        let mut count_total = 0;

        for item in self.non_zero_buckets() {
            let count = item.count;
            let tag = format!("vmrange={}", item.vm_range);
            let metric_name = add_tag(prefix, tag);
            let (name, labels) = split_metric_name(&metric_name);
            dst.push_str(format!("{}_bucket${} {}\n", name, labels, count).as_str());
            count_total += count;
        }

        if count_total == 0 {
            return;
        }
        let (name, labels) = split_metric_name(prefix);
        let sum = self.sum;
        if sum.floor() == sum {
            let msg = format!("{}_sum{} {}\n", name, labels, sum.floor());
            dst.push_str(&*msg);
        } else {
            let msg = format!("{}_sum{} {}\n", name, labels, sum);
            dst.push_str(&*msg);
        }
        dst.push_str(&*format!("{}_count{} {}\n", name, labels, count_total));
    }

    /// Get an iterator over this histogram's buckets.
    pub fn non_zero_buckets(&self) -> NonZeroBuckets {
        NonZeroBuckets {
            histogram: self,
            index: 0,
            offset: 0,
            stage: IterationStage::Lower,
        }
    }

    pub fn quantile(&self, phi: f64) -> f64 {
        crate::functions::quantile(phi, &self.values)
    }
}

/// Iterator return type.
#[derive(Clone)]
pub struct NonZeroBucket<'a> {
    pub vm_range: &'a str,
    /// The number of samples in this bucket's range.
    pub count: u64,
}

impl<'a> fmt::Debug for NonZeroBucket<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bucket {{ {}..{} }}", self.vm_range, self.count)
    }
}

#[derive(PartialEq, Clone, Debug)]
enum IterationStage {
    Lower,
    Main,
    Higher,
    Done,
}

impl Display for IterationStage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IterationStage::Lower => write!(f, "Lower"),
            IterationStage::Main => write!(f, "Main"),
            IterationStage::Higher => write!(f, "Higher"),
            IterationStage::Done => write!(f, "Done"),
        }
    }
}

/// An iterator over the non-zero buckets in a histogram.
#[derive(Debug, Clone)]
pub struct NonZeroBuckets<'a> {
    histogram: &'a Histogram,
    index: usize,
    offset: usize,
    stage: IterationStage,
}

impl<'a> Iterator for NonZeroBuckets<'a> {
    type Item = NonZeroBucket<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stage == IterationStage::Lower {
            self.stage = IterationStage::Main;
            if self.histogram.lower > 0 {
                return Some(NonZeroBucket {
                    vm_range: &LOWER_BUCKET_RANGE,
                    count: self.histogram.lower,
                });
            }
        }

        if self.stage == IterationStage::Higher {
            self.stage = IterationStage::Done;
            if self.histogram.upper > 0 {
                return Some(NonZeroBucket {
                    vm_range: &UPPER_BUCKET_RANGE,
                    count: self.histogram.upper,
                });
            }
            return None;
        }

        let buckets = &self.histogram.decimal_buckets;

        let result = loop {
            while self.index < buckets.len() && buckets[self.index].is_empty() {
                self.index += 1;
            }

            if self.index >= buckets.len() {
                self.stage = IterationStage::Done;
                break None;
            }

            let bucket = &buckets[self.index];

            while self.offset < bucket.len() && bucket[self.offset] == 0 {
                self.offset += 1
            }

            if self.offset >= bucket.len() {
                self.index += 1;
                self.offset = 0;
                continue;
            }

            let bucket_idx = self.index * BUCKETS_PER_DECIMAL + self.offset;
            let ranges = &*BUCKET_RANGES;
            let vm_range = &ranges[bucket_idx];
            let count = bucket[self.offset];

            break Some(NonZeroBucket { vm_range, count });
        };

        result
    }
}

#[inline]
fn format_float(v: f64) -> String {
    format!("{:.3}", v)
}

static BUCKET_RANGES: Lazy<Vec<String>> = Lazy::new(|| {
    let mut ranges: Vec<String> = Vec::with_capacity(BUCKETS_COUNT);
    let mut v: f64 = LOWER_MIN;
    let mut start = format_float(v);
    let mut i = 0;
    while i < BUCKETS_COUNT {
        v *= BUCKET_MULTIPLIER;
        let end = format_float(v);
        ranges.push(format!("{}...{}", start, end).to_string());
        start = end;
        i += 1;
    }
    ranges
});

// todo: move to utils ?
// todo - return slices instead
fn split_metric_name(name: &str) -> (&str, &str) {
    if let Some(index) = name.find('{') {
        if let Some(tailing_index) = name.find('}') {
            let metric_name = &name[1..index];
            let labels = &name[index + 1..tailing_index];
            return (metric_name, labels);
        }
    }
    (name, "")
}

fn add_tag(name: &str, tag: String) -> String {
    let need_braces = if name.is_empty() {
        true
    } else {
        let last_ch = name.chars().last().unwrap();
        last_ch == '}'
    };

    if need_braces {
        return format!("{}{{{}}}", name, tag);
    }
    let mut chars = name.chars();
    chars.next_back();
    format!("{},{}}}", chars.as_str(), tag)
}

fn get_pool() -> &'static LinearObjectPool<Histogram> {
    static INSTANCE: OnceCell<LinearObjectPool<Histogram>> = OnceCell::new();
    INSTANCE.get_or_init(|| LinearObjectPool::<Histogram>::new(Histogram::new, |v| v.reset()))
}

pub fn get_pooled_histogram() -> LinearReusable<'static, Histogram> {
    get_pool().pull()
}
