

pub fn round_to_seconds(ms: i64) -> i64 {
    return ms - &ms % 1000;
}

// These values prevent from overflow when storing msec-precision time in int64.
const MIN_TIME_MSECS: i64  = 0; // use 0 instead of `int64(-1<<63) / 1e6` because the storage engine doesn't actually support negative time
const MAX_TIME_MSECS: i64 = (1<<63-1) / 1e6;

const maxDurationMsecs: i64 = 100 * 365 * 24 * 3600 * 1000;


// Deadline contains deadline with the corresponding timeout for pretty error messages.
pub struct Deadline {
    deadline: u64,
    timeout:  i64,
    flag_hint: String
}

// new_deadline returns deadline for the given timeout.
//
// flag_hint must contain a hit for command-line flag, which could be used
// in order to increase timeout.
fn new_deadline(start_time: time.Time, timeout: time.Duration, flag_hint: String) -> Deadline {
    return Deadline {
        deadline: uint64(start_time.Add(timeout).Unix()),
        timeout,
        flagHint: flag_hint,
    }
}

impl Deadline {
    // Exceeded returns true if deadline is exceeded.
    pub fn exceeded(self) {
        return fasttime.UnixTimestamp() > self.deadline        
    }

    // Deadline returns deadline in unix timestamp seconds.
    pub fn deadline(&self) {
        return self.deadline;
    }
}


// String returns human-readable string representation for d.
func (d *Deadline) String() string {
startTime := time.Unix(int64(d.deadline), 0).Add(-d.timeout)
elapsed := time.Since(startTime)
return fmt.Sprintf("%.3f seconds (elapsed %.3f seconds); the timeout can be adjusted with `%s` command-line flag", d.timeout.Seconds(), elapsed.Seconds(), d.flagHint)
}


// join_tag_filterss adds etfs to every src filter and returns the result.
fn join_tag_filterss(src: Vec<Vec<TagFilter>>, etfs: Vec<Vec<TagFilter>>) -> Vec<Vec<TagFilter>> {
    if src.len() == 0 {
        return etfs
    }
    if etfs.len() == 0 {
        return src
    }
    let dst: Vec<Vec<TagFilter>> = Vec::with_capacity(src.len());
    for tf in src {
        for etf in etfs {
            let tfs: Vec<TagFilter> = tf.iter().cloned();
            tfs.append(etf.iter().cloned());
            dst.push(tfs);
        }
    }
    return dst;
}

// ParseMetricSelector parses s containing PromQL metric selector and returns the corresponding LabelFilters.
fn parse_metric_selector(s: String) -> Result<Vec<TagFilter>, Error> {
    let expr = metricsql.Parse(s);
    return match expr {
        Expression::MetricExpression(me) => {
            if me.LabelFilters == 0 {
                let msg = "labelFilters cannot be empty";
                return Err(Error::new(msg));
            }
            to_tag_filters(me.label_filters)
        },
        _ => {
            let msg = format!("expecting metricSelector; got {}", expr);
            Err(Error::new(msg))
        }
    }
}

// ToTagFilters converts lfs to a slice of storage.TagFilter
fn to_tag_filters(lfs: &[LabelFilter]) -> Vec<TagFilter> {
    let tfs: Vec<TagFilter> = Vec::with_capacity(lfs.len());
    for i in 0 .. lfs.len() {
        to_tag_filter(&tfs[i], &lfs[i])
    }
    return tfs
}

pub fn to_tag_filter(mut dst: &TagFilter, src: &LabelFilter) {
    if src.Label != "__name__" {
        dst.Key = []byte(src.Label)
    } else {
        // This is required for storage.Search.
        dst.Key = nil
    }
    dst.value = []byte(src.Value)
    dst.is_regexp = src.is_regexp;
    dst.is_negative = src.IsNegative
}