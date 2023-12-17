use std::fmt;
use std::fmt::Display;
use std::ops::Add;
use std::time::Duration;

use crate::types::{Timestamp, TimestampTrait};
use crate::{RuntimeError, RuntimeResult};

/// These values prevent from overflow when storing ms-precision time in i64.
const MIN_TIME_MSECS: i64 = 0;
pub const MAX_DURATION_MSECS: i64 = 100 * 365 * 24 * 3600 * 1000;

/// Deadline contains deadline with the corresponding timeout for pretty error messages.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Deadline {
    /// deadline in unix timestamp seconds.
    pub deadline: Timestamp,
    pub timeout: Duration,
}

impl Deadline {
    /// Returns a deadline for the given timeout.
    pub fn new(timeout: Duration) -> RuntimeResult<Self> {
        Deadline::with_start_time(Timestamp::now(), timeout)
    }

    pub fn from_now(timeout: Duration) -> RuntimeResult<Self> {
        Deadline::with_start_time(Timestamp::now(), timeout)
    }

    /// Returns a deadline for the given start time and timeout.
    pub fn with_start_time<T>(start_time: T, timeout: Duration) -> RuntimeResult<Self>
    where
        T: Into<Timestamp>,
    {
        let millis = timeout.as_millis() as i64;
        if millis > MAX_DURATION_MSECS {
            return Err(RuntimeError::ArgumentError(format!(
                "Timeout value too large: {:?}",
                timeout
            )));
        }
        if millis < MIN_TIME_MSECS {
            return Err(RuntimeError::ArgumentError(format!(
                "Negative timeouts are not supported. Got {:?}",
                timeout
            )));
        }
        Ok(Deadline {
            deadline: start_time.into().add(timeout.as_millis() as i64),
            timeout,
        })
    }

    /// returns true if deadline is exceeded.
    pub fn exceeded(&self) -> bool {
        Timestamp::now() > self.deadline
    }
}

impl Default for Deadline {
    fn default() -> Self {
        let start = Timestamp::now();
        let timeout = Duration::from_secs(10); // todo: constant
        Deadline {
            deadline: start.add(timeout.as_millis() as i64),
            timeout,
        }
    }
}

impl TryFrom<Duration> for Deadline {
    type Error = RuntimeError;

    fn try_from(timeout: Duration) -> Result<Self, Self::Error> {
        Deadline::new(timeout)
    }
}

impl TryFrom<u64> for Deadline {
    type Error = RuntimeError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let timeout = Duration::from_secs(value);
        Deadline::new(timeout)
    }
}

impl Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start_time = self.deadline - self.timeout.as_millis() as i64;
        let elapsed = (Timestamp::now() - start_time) / 1000_i64;
        write!(
            f,
            "{:.3} seconds (elapsed {:.3} seconds);",
            self.timeout.as_secs(),
            elapsed
        )?;
        Ok(())
    }
}
