use std::fmt::Display;
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Enum for various alert errors.
#[derive(Debug, Clone, Error, Eq, PartialEq, Serialize, Deserialize)]
pub enum RelabelError {
    #[error("Invalid configuration. {0}")]
    InvalidConfiguration(String),

    #[error("Invalid rule. {0}")]
    InvalidRule(String),

    #[error("Configuration error. {0}")]
    Configuration(String),

    #[error("Serialization error. {0}")]
    CannotSerialize(String),

    #[error("Cannot deserialize. {0}")]
    CannotDeserialize(String),

    #[error("Duplicate series. {0}")] // need better error
    DuplicateSeries(String),

    #[error("Invalid series selector: {0}")]
    InvalidSeriesSelector(String),

    #[error("Failed to execute query: {0}")]
    QueryExecutionError(String),

    #[error("Failed to create alert. {0}")]
    FailedToCreateAlert(String),

    #[error("{0}")]
    Generic(String),

    #[error("Failure restoring rule: {0}")]
    RuleRestoreError(String)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct ErrorGroup(pub Vec<String>);

impl ErrorGroup {
    pub fn new() -> Self {
        ErrorGroup(Vec::new())
    }

    pub fn push(&mut self, err: String) {
        self.0.push(err);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Display for ErrorGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.join(", "))
    }
}

impl FromStr for ErrorGroup {
    type Err = RelabelError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ErrorGroup(vec![s.to_string()]))
    }
}

impl From<ErrorGroup> for RelabelError {
    fn from(err: ErrorGroup) -> Self {
        RelabelError::Generic(err.0.join(", "))
    }
}

pub struct MaxActivePendingExceeded {
    pub max_active: usize,
    pub limit: usize,
}

pub type RelabelResult<T> = Result<T, RelabelError>;