use metricsql_parser::prelude::ParseError;
use thiserror::Error;

/// Enum for various Relabel errors.
#[derive(Debug, Clone, Error, Eq, PartialEq)]
pub enum RelabelError {
    #[error("Invalid configuration. {0}")]
    InvalidConfiguration(String),

    #[error("Invalid rule. {0}")]
    InvalidRule(String),

    #[error("Duplicate series. {0}")] // need better error
    DuplicateSeries(String),

    #[error("Invalid series selector: {0}")]
    InvalidSeriesSelector(String),

    #[error("{0}")]
    Generic(String),

    #[error("Error parsing config: {:?}", .0)]
    ParseError(ParseError)
}


pub type RelabelResult<T> = Result<T, RelabelError>;