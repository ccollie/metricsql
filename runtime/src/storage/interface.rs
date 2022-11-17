// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// The errors exposed.

const ERR_NOT_FOUND: &str = "not found";
const ERR_OUT_OF_ORDER_SAMPLE: &str = "out of order sample";
const ERR_DUPLICATE_SAMPLE_FOR_TIMESTAMP: &str = "duplicate sample for timestamp";
const ERR_OUT_OF_BOUNDS: &str = "out of bounds";


/// SeriesRef is a generic series reference. In prometheus it is either a
/// HeadSeriesRef or BlockSeriesRef, though other implementations may have
/// their own reference types.
pub type SeriesRef = u64;

/// Appendable allows creating appenders.
pub trait Appendable {
    /// Appender returns a new appender for the storage. The implementation
    /// can choose whether or not to use the context, for deadlines or to check
    /// for errors.
    fn get_appender(ctx: Context) -> dyn Appender;  // todo: make this Arc??
}

/// SampleAndChunkQueryable allows retrieving samples as well as encoded samples in form of chunks.
pub trait SampleAndChunkQueryable: Queryable + ChunkQueryable  {
}

/// Storage ingests and manages samples, along with various indexes. All methods
/// are goroutine-safe. Storage implements storage.Appender.
pub trait Storage: Appendable {
    SampleAndChunkQueryable
    
    /// StartTime returns the oldest timestamp stored in the storage.
    fn start_time(&self) -> RuntimeResult<i64>;

    /// Close closes the storage and all its underlying resources.
    fn close(&mut self) -> RuntimeResult<()>;
}

/// A Queryable handles queries against a storage.
/// Use it when you need to have access to all samples without chunk encoding abstraction e.g promQL.
pub trait Queryable {
    // Querier returns a new Querier on the storage.
    fn get_querier(ctx:Context, mint: i64, maxt: i64) -> RuntimeResult<Querier>;
}

/// A MockQueryable is used for testing purposes so that a mock Querier can be used.
pub struct MockQueryable {
    mock_querier: Querier
}

impl Queryable for MockQueryable {
    fn get_querier(ctx: Context, mint: i64, maxt: i64) -> RuntimeResult<Querier> {
        todo!()
    }    
}

/// Querier provides querying access over time series data of a fixed time range.
pub trait Querier: LabelQuerier {
    /// Select returns a set of series that matches the given label matchers.
    /// Caller can specify if it requires returned series to be sorted. Prefer not requiring sorting
    /// for better performance. It allows passing hints that can help in optimising select, but it's
    /// up to implementation how this is used if used at all.
    fn select(&self, sort_series: bool, hints: &SelectHints, matchers: &[Matcher]) -> SeriesSet;
}

// MockQuerier is used for test purposes to mock the selected series that is returned.
pub struct MockQuerier {
    select_function: fn(sort_series: bool, hints: &SelectHints, matchers: &[Matcher]) -> SeriesSet;
}

impl LabelQuerier for MockQuerier {
    fn label_values(&self, name: string, matchers: &[Matcher]) -> RuntimeResult<(Vec<String>, Warnings)> {
        return Ok((vec![], vec![]));
    }

    fn label_names(&self, name: string, matchers: &[Matcher]) -> RuntimeResult<(Vec<String>, Warnings)> {
        return Ok((vec![], vec![]))
    }
}

impl MockQuerier {
    
    fn close(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    fn select(&self, sort_series: bool, hints: &SelectHints, matchers: &[Matcher]) -> SeriesSet {
        (self.select_function)(sort_series, hints, matchers)
    }
}


/// A ChunkQueryable handles queries against a storage.
/// Use it when you need to have access to samples in encoded format.
pub trait ChunkQueryable {
    // chunk_querier returns a new chunk_querier on the storage.
    fn chunk_querier(ctx: Context, mint: i64, maxt: i64) -> RuntimeResult<ChunkQuerier>;
}

/// chunk_querier provides querying access over time series data of a fixed time range.
pub trait ChunkQuerier: LabelQuerier {
    /// Select returns a set of series that matches the given label matchers.
    /// Caller can specify if it requires returned series to be sorted. Prefer not requiring sorting for better performance.
    /// It allows passing hints that can help in optimising select, but it's up to implementation how this is used if used at all.
    fn select(&self, sort_series: bool, hints: &SelectHints, matchers: &[Matcher]) -> RuntimeResult<ChunkSeriesSet>;
}

/// LabelQuerier provides querying access over labels.
pub trait LabelQuerier {
    /// label_values returns all potential values for a label name.
    /// It is not safe to use the strings beyond the lifetime of the querier.
    /// If matchers are specified the returned result set is reduced
    /// to label values of metrics matching the matchers.
    fn label_values(&self, name: string, matchers: &[Matcher]) -> RuntimeResult<Vec<String>>;

    /// LabelNames returns all the unique label names present in the block in sorted order.
    /// If matchers are specified the returned result set is reduced
    /// to label names of metrics matching the matchers.
    fn label_names(&self, matchers: &[Matcher])  -> RuntimeResult<Vec<String>>;

    /// Close releases the resources of the Querier.
    fn close(&mut self) -> RuntimeResult<()>;
}

/// SelectHints specifies hints passed for data selections.
/// This is used only as an option for implementation to use.
pub struct SelectHints {
    start: i64, // start time in milliseconds for this select.
    end: i64, // end time in milliseconds for this select.

    step: i64,  // Query step size in milliseconds.
    func: String, // String representation of surrounding function or aggregation.

    grouping: Vec<String>, // List of label names used in aggregation.
    by:       bool,     // Indicate whether it is without or by.
    range   : i64,    // Range vector selector range in milliseconds.

    // disable_trimming allows to disable trimming of matching series chunks based on query start and end time.
    // When disabled, the result may contain samples outside the queried time range but Select() performances
    // may be improved.
    disable_trimming: bool
}

/// QueryableFunc is an adapter to allow the use of ordinary functions as
/// Queryables. It follows the idea of http.HandlerFunc.
type QueryableFunc = fn(ctx: Context, mint: i64, maxt: i64) -> RuntimeResult<Querier>;


/// Appender provides batched appends against a storage.
/// It must be completed with a call to `commit` or `rollback` and must not be reused afterwards.
///
/// Operations on the Appender interface are not thread-safe.
pub trait Appender {
    /// append adds a sample pair for the given series.
    /// An optional series reference can be provided to accelerate calls.
    /// A series reference number is returned which can be used to add further
    /// samples to the given series in the same or later transactions.
    /// Returned reference numbers are ephemeral and may be rejected in calls
    /// to `append()` at any point. Adding the sample via `append()` returns a new
    /// reference number.
    /// If the reference is 0 it must not be used for caching.
    fn append(&mut self, sref: SeriesRef, l: Labels, t: i64, v: f64) -> RuntimeResult<SeriesRef>;

    /// Commit submits the collected samples and purges the batch. If Commit
    /// returns a non-nil error, it also rolls back all modifications made in
    /// the appender so far, as Rollback would do. In any case, an Appender
    /// must not be used anymore after Commit has been called.
    fn commit(&mut self) -> RuntimeResult<()>;

    /// Rollback rolls back all modifications made in the appender so far.
    /// Appender has to be discarded after rollback.
    fn rollback(&mut self) -> RuntimeResult<()>;
}

/// GetRef is an extra interface on Appenders used by downstream projects
/// (e.g. Cortex) to avoid maintaining a parallel set of references.
pub trait GetRef {
    /// Returns reference number that can be used to pass to Appender.append(),
    /// and a set of labels that will not cause another copy when passed to Appender.append().
    /// 0 means the appender does not have a reference to this series.
    fn get_ref(self, lset: &Labels) -> (SeriesRef, Labels);
}

/// SeriesSet contains a set of series.
pub trait SeriesSet {
    fn next(&mut self) -> Option<Series> {
        false
    }
    /// A collection of warnings for the whole set.
    /// Warnings could be return even iteration has not failed with error.
    fn warnings(&self) -> Warnings;
}

pub struct EmptySeriesSet{}

impl EmptySeriesSet {
    /// EmptySeriesSet returns a series set that's always empty.
    pub fn new() -> Self {
        Self {}
    }
}

impl SeriesSet for EmptySeriesSet {
    fn next(&mut self) -> Option<Series> {
        None
    }

    fn warnings(&self) -> Warnings {
        todo!()
    }
}

pub struct TestSeriesSet  {
    pub(crate) series: Series
}

impl TestSeriesSet {
    pub fn new(series: Series) -> Self {
        return TestSeriesSet { series }
    }
}

impl SeriesSet for TestSeriesSet {
    fn next(&mut self) -> Option<Series> {
        None
    }
    fn warnings(&self) -> Warnings {
        todo!()
    }
}

/// ErrSeriesSet returns a series set that wraps an error.
pub struct ErrSeriesSet {
    err: RuntimeError
}

impl ErrSeriesSet {
    pub fn new(err: RuntimeError) -> Self {
        Self {
            err
        }
    }
}

impl SeriesSet for ErrSeriesSet {
    fn next(&mut self) -> Option<Series> {
        None
    }

    fn warnings(&self) -> Warnings {
        todo!()
    }
}


#[derive(Default)]
pub struct ErrChunkSeriesSet {
    err: RuntimeError
}

impl EmptySeriesSet {
    pub fn new(err: RuntimeError) -> Self {
        return ErrChunkSeriesSet{ err }
    }
}

impl SeriesSet for ErrSeriesSet {
    fn next(&mut self) -> Option<Series> {
        None
    }
}


// Series exposes a single time series and allows iterating over samples.
pub trait Series {
    fn labels(&self) -> MetricName;
    SampleIterable
}

#[derive(Default, Clone)]
pub struct MockSeries {
    timestamps: Vec<i64>,
    values:     Vec<f64>,
    label_set: Vec<String>
}

impl MockSeries {
    /// returns a series with custom timestamps, values and label_set.
    pub fn new(timestamps: &[i64], values: &[f64], label_set: Vec<String>) -> Self {
        return MockSeries{
            timestamps: Vec::from(timestamps),
            values:     Vec::from(values),
            label_set:   label_set.clone(),
        }
    }

    pub fn labels(&self) -> MetricName {
        return MetricName::from_strings(self.label_set)
    }
}


/// ChunkSeriesSet contains a set of chunked series.
pub trait ChunkSeriesSet {
    fn next(self) -> Option<ChunkSeries>;
    // A collection of warnings for the whole set.
    // Warnings could be return even iteration has not failed with error.
    fn warnings(self) -> Warnings;
}

/// ChunkSeries exposes a single time series and allows iterating over chunks.
pub trait ChunkSeries {
Labels
ChunkIterable
}

// Labels represents an item that has labels e.g. time series.
pub trait Labels {
    // Labels returns the complete set of labels. For series it means all labels identifying the series.
    fn labels(&self) -> Labels
}

pub trait SampleIterable {
    // Iterator returns a new, independent iterator of the data of the series.
    Iterator() chunkenc.Iterator
}

pub trait ChunkIterable {
    /// Iterator returns a new, independent iterator that iterates over potentially overlapping
    /// chunks of the series, sorted by min time.
    Iterator() chunks.Iterator
}

pub type Warnings = Vec<RuntimeError>;