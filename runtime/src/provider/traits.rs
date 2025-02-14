use super::error::ProviderResult;
use crate::types::Sample;
use async_trait::async_trait;
use metricsql_parser::label::{Labels, Matcher};
use crate::SeriesRef;

pub struct AppendOptions {
    discard_out_of_order: bool
}

/// Appender provides batched appends against a storage.
/// It must be completed with a call to commit or rollback.
///
///
/// The type of samples (float64, histogram, etc) appended for a given series must remain same within an Appender.
/// The behaviour is undefined if samples of different types are appended to the same series in a single Commit().
pub trait Appender {
    /// `append` adds a sample pair for the given series.
    /// An optional series reference can be provided to accelerate calls.
    /// A series reference number is returned which can be used to add further
    /// samples to the given series in the same or later transactions.
    /// Adding the sample via `append()` returns a new reference number.
    /// If the reference is 0 it must not be used for caching.
    async fn append(&mut self, sref: SeriesRef, labels: Labels, ts: i64, value: f64) -> ProviderResult<SeriesRef>;

    /// `commit()` submits the collected samples and purges the batch. If `commit()`
    /// returns an Err, it also rolls back all modifications made in
    /// the appender so far, as `rollback()` would do.
    async fn commit(&mut self) -> ProviderResult;

    /// `rollback` rolls back all modifications made in the appender so far.
    async fn rollback(&mut self) -> ProviderResult;

    /// `set_options` configures the appender with specific append options such as
    /// discarding out-of-order samples even if out-of-order is enabled in the TSDB.
    fn set_options(&mut self, opts: &AppendOptions);
}

/// `LabelHints` specifies hints passed for label reads.
/// This is used only as an option for implementation to use.
pub struct LabelHints {
    // Maximum number of results returned. Use a value of 0 to disable.
    limit: usize
}

// LabelQuerier provides querying access over labels.
#[async_trait]
pub trait LabelQuerier {
    /// `label_values` returns all potential values for a label name in sorted order.
    /// If matchers are specified the returned result set is reduced
    /// to label values of metrics matching the matchers.
    async fn label_values(&self, name: &str, hints: &LabelHints, matchers: &[Matcher]) -> ProviderResult<Vec<String>>;

    /// `label_names` returns all the unique label names present in the block in sorted order.
    /// If matchers are specified the returned result set is reduced to label names of metrics matching the matchers.
    async fn label_names(&self, name: &str, hints: &LabelHints, matchers: &[Matcher]) -> ProviderResult<Vec<String>>;
}

/// SelectHints specifies hints passed for data selections.
/// This is used only as an option for implementation to use.
pub struct SelectOptions {
    /// Start time in milliseconds for this select.
    start: i64,
    /// End time in milliseconds for this select.
    end: i64,

    /// Maximum number of results returned. Use a value of 0 to disable.
    limit: i32,

    /// Query step size in milliseconds.
    step: i64,
    
    // Specify if returned series are to be sorted. Prefer not requiring sorting for better performance.
    sort_series: bool
}

/// Series exposes a single time series and allows iterating over samples.
pub trait Series: Sized {
    /// Labels returns the complete set of labels. For series it means all labels identifying the series.
    fn labels(&self) -> Labels;

    /// Iterator returns an iterator of the data of the series.
    fn iterator(&self) -> impl Iterator<Item=Sample>;
}

/// Querier provides querying access over time series data of a fixed time range.
pub trait Querier<S: Series>: LabelQuerier + Send + Sync {
    /// `select` returns a set of series that matches the given label matchers.
    /// Results are not checked whether they match. Results that do not match may cause undefined behavior.
    /// It allows passing hints that can help in optimising select, but it's up to implementation how 
    /// this is used if used at all.
    async fn select(&self, hints: &SelectOptions, matchers: &[Matcher]) -> ProviderResult<Vec<S>>;
}
