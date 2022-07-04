use std::ops::Range;

pub type TimeRange = Range<i64>;

// Search is a search for time series.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Search {
    // tr contains time range used in the search.
    tr: TimeRange,

    // tfss contains tag filters used in the search.
    tfss: Vec<TagFilter>,

    // deadline in unix timestamp seconds for the current search.
    deadline: u64,

    err: Error,

    need_closing: bool,
}

impl Search {
    pub fn reset(&mut self) {
        self.tr = TimeRange{ start: 0, end: i64::MAX };
        self.tfss = vec![];
        self.deadline = 0;
        self.err = nil;
        self.need_closing = false;
    }
}

// Init initializes s from the given storage, tfss and tr.
//
// MustClose must be called when the search is done.
//
// Init returns the upper bound on the number of found time series.
func (s *Search) Init(
qt: QueryTracer, storage: Storage, tfss: &[TagFilters], tr: &TimeRange, maxMetrics int, deadline: u64) int {
qt = qt.NewChild("init series search: filters=%s, timeRange=%s", tfss, &tr)
defer qt.Done()
if s.needClosing {
logger.Panicf("BUG: missing MustClose call before the next call to Init")
}

}

// Error returns the last error from s.
func (s *Search) Error() error {
if s.err == io.EOF || s.err == nil {
return nil
}
return fmt.Errorf("error when searching for tagFilters=%s on the time range %s: %w", s.tfss, s.tr.String(), s.err)
}

// SearchQuery is used for sending search queries from vmselect to storage.
pub struct SearchQuery {
    // The time range for searching time series
    min_timestamp: i64,
    max_timestamp: i64,

    // Tag filters for the search query
    tag_filterss: Vec<Vec<TagFilter>>,

    // The maximum number of time series the search query can return.
    max_metrics: usize
}

impl SearchQuery {
    // NewSearchQuery creates new search query for the given args.
    pub fn new(start: i64, end: i64, tag_filterss: Vec<Vec<TagFilter>>, max_metrics: i32) -> Self {
        if max_metrics <= 0 {
            max_metrics = 2e9
        }
        SearchQuery{
            min_timestamp: start,
            max_timestamp: end,
            tag_filterss: tag_filterss,
            max_metrics,
        }
    }
}

impl Display for SearchQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        write!(f, "{}", self.op)?;
        if !self.labels.is_empty() {
            write!(f, " ")?;
            write_labels(&self.labels, f);
        }

        let a = make([]string, len(sq.TagFilterss))
        for i, tfs := range sq.TagFilterss {
            a[i] = tagFiltersToString(tfs)
        }
        start = TimestampToHumanReadableFormat(sq.MinTimestamp)
        end = TimestampToHumanReadableFormat(sq.MaxTimestamp)
        return fmt.Sprintf("filters=%s, timeRange=[%s..%s]", a, start, end)

        Ok(())
    }
}

// TagFilter represents a single tag filter from SearchQuery.
pub struct  TagFilter {
    key        []byte
    Value      []byte
    is_negative: bool
    is_regexp:   bool
}

impl TagFilter {
    pub fn get_op(self) {
        if self.is_negative {
            if self.is_regexp {
                return "!~"
            }
            return "!="
        }
        if self.is_regexp {
            return "=~"
        }
        return "="
    }
}

impl Display for TagFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let op = self.getOp();
        if self.key == "__name__" || self.key.len == 0 {
            write!(f, "__name__{}{}", op, value);
        } else {
            write!(f, "{}{}{}", self.key, op, self.value);
        }
        Ok(())
    }
}


fn tag_filters_to_string(tfs: &[TagFilter]) -> String {
    let a = make([]string, len(tfs))
    for tf in tfs {
        a[i] = tf.String()
    }
    return "{" + strings.Join(a, ",") + "}"
}



// Result is a single timeseries result.
//
// ProcessSearchQuery returns Result slice.
pub struct QueryResult<'a> {
    // The name of the metric.
    metric_name: MetricName,
    // Values are sorted by Timestamps.
    values: &'a  &[f64],
    timestamps: &'a &[i64],
}

impl QueryResult<'a> {
    pub fn reset(&mut self) {
        self.metric_name.reset();
        self.values.clear();
        self.timestamps.clear();
    }
}

// Results holds results returned from ProcessSearchQuery.
pub struct Results {
    tr: TimeRange,
    deadline: Deadline,
    series: Vec<QueryResult>,
    sr: Search,
}

impl Results {
    // Len returns the number of results in rss.
    pub fn len(self) -> usize {
        self.series.len()
    }
}


var resultPool sync.Pool


