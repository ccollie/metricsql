use thiserror::Error;
use crate::SeriesRef;

/// Alias for a type-erased error type.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub type ProviderResult<T=()> = Result<T, ProviderError>;

#[derive(Debug, PartialEq, Clone, Error)]
pub enum ProviderError {
    #[error("Invalid matcher: `{0}`")]
    InvalidMatcher(String),
    #[error("Missing matching filter")]
    MissingMatcher,
    #[error("Error fetching postings")]
    PostingFetchError,
    #[error("Missing posting in index")]
    MissingPostingInIndex,
    #[error("Duplicate posting in index for metric: \"{0}\"")]
    DuplicatePostingInIndex(String),
    #[error("{0}")]
    General(String),
    #[error("Deadline exceeded: {0}")]
    DeadlineExceededError(String),
    #[error("Task cancelled: {0}")]
    TaskCancelledError(String),
    #[error("Duplicate output series: {0}")]
    DuplicateOutputSeries(String),
    #[error("Posting serialization error: {0}")]
    PostingSerializationError(String),
    #[error("The response contains more than {max_series} series: found {found_series};")]
    MaxSeriesExceeded {
        found_series: usize,
        max_series: usize,
    },
    #[error("Series not found for ID: {0}")]
    SeriesNotFound(SeriesRef),
    #[error("No postings found for label: {0}")]
    NoPostingsForLabel(String),
    #[error("No postings found for label value: {0}")]
    NoPostingsForLabelValue(String),
    // IndexNotFound,
    #[error("Invalid index or index not found")]
    InvalidIndex,
    // // CorruptedIndex,
    #[error("Internal error \"{reason}\": Source {source:?}")]
    Internal {
        reason: String,
        source: Option<BoxError>,
    },
}

impl ProviderError {
    pub fn deadline_exceeded(s: &str) -> Self {
        ProviderError::DeadlineExceededError(s.to_string())
    }
}

impl From<&str> for ProviderError {
    fn from(message: &str) -> Self {
        ProviderError::General(String::from(message))
    }
}

impl From<String> for ProviderError {
    fn from(message: String) -> Self {
        ProviderError::General(message)
    }
}

impl<E: std::error::Error + 'static> From<(String, E)> for ProviderError {
    fn from((message, err): (String, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        ProviderError::General(msg)
    }
}

impl<E: std::error::Error + 'static> From<(&str, E)> for ProviderError {
    fn from((message, err): (&str, E)) -> Self {
        let msg = format!("{}: {}", message, err);
        ProviderError::General(msg)
    }
}


