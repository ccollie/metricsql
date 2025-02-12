use std::error::Error;
use async_trait::async_trait;
use metricsql_parser::label::{Labels, Matcher};
use crate::RuntimeResult;
use crate::types::Sample;

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
    async fn label_values(&self, name: &str, hints: &LabelHints, matchers: &[Matcher]) -> Result<Vec<String>, Box<dyn Error>>;

    /// `label_names` returns all the unique label names present in the block in sorted order.
    /// If matchers are specified the returned result set is reduced to label names of metrics matching the matchers.
    async fn label_names(&self, name: &str, hints: &LabelHints, matchers: &[Matcher]) -> Result<Vec<String>, Box<dyn Error>>;
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
pub trait Querier: LabelQuerier + Send + Sync {
    /// `select` returns a set of series that matches the given label matchers.
    /// Results are not checked whether they match. Results that do not match may cause undefined behavior.
    /// It allows passing hints that can help in optimising select, but it's up to implementation how this is used if used at all.
    async fn select(&self, hints: &SelectOptions, matchers: &[Matcher]) -> RuntimeResult<Vec<dyn Series>>;
}
