use std::borrow::Cow;
use std::fmt;
use std::fmt::Display;
use std::ops::Add;

use chrono::Duration;
use metricsql::ast::Expr;
use metricsql::common::LabelFilter;
use metricsql::parser::parse;

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::{Timestamp, TimestampTrait};

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

    /// Returns a deadline for the given start time and timeout.
    pub fn with_start_time<T>(start_time: T, timeout: Duration) -> RuntimeResult<Self>
    where
        T: Into<Timestamp>,
    {
        let millis = timeout.num_milliseconds();
        if millis > MAX_DURATION_MSECS {
            return Err(RuntimeError::ArgumentError(format!(
                "Timeout value too large: {}",
                timeout
            )));
        }
        if millis < MIN_TIME_MSECS {
            return Err(RuntimeError::ArgumentError(format!(
                "Negative timeouts are not supported. Got {}",
                timeout
            )));
        }
        return Ok(Deadline {
            deadline: start_time.into().add(timeout.num_milliseconds()),
            timeout,
        });
    }

    /// returns true if deadline is exceeded.
    pub fn exceeded(&self) -> bool {
        Timestamp::now() > self.deadline
    }
}

impl Default for Deadline {
    fn default() -> Self {
        let start = Timestamp::now();
        let timeout = Duration::seconds(10); // todo: constant
        Deadline {
            deadline: start.add(timeout.num_milliseconds()),
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

impl TryFrom<i64> for Deadline {
    type Error = RuntimeError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let timeout = Duration::milliseconds(value);
        Deadline::new(timeout)
    }
}

impl Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start_time = self.deadline - self.timeout.num_milliseconds();
        let elapsed = (Timestamp::now() - start_time) / 1000_i64;
        write!(
            f,
            "{:.3} seconds (elapsed {:.3} seconds);",
            self.timeout.num_seconds(),
            elapsed
        )?;
        Ok(())
    }
}

/// join_tag_filter_list adds etfs to every src filter and returns the result.
pub(crate) fn join_tag_filter_list<'a>(
    src: &'a Vec<Vec<LabelFilter>>,
    etfs: &'a Vec<Vec<LabelFilter>>,
) -> Cow<'a, Vec<Vec<LabelFilter>>> {
    if src.len() == 0 {
        return Cow::Borrowed::<'a>(etfs);
    }
    if etfs.len() == 0 {
        return Cow::Borrowed::<'a>(src);
    }
    let mut dst: Vec<Vec<LabelFilter>> = Vec::with_capacity(src.len());
    for tf in src.iter() {
        for etf in etfs.iter() {
            let mut tfs: Vec<LabelFilter> = tf.clone();
            tfs.append(&mut etf.clone());
            dst.push(tfs.into());
        }
    }
    Cow::Owned::<'a>(dst)
}

/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RuntimeResult<Vec<LabelFilter>> {
    return match parse(s) {
        Ok(expr) => match expr {
            Expr::MetricExpression(me) => {
                if me.is_empty() {
                    let msg = "labelFilters cannot be empty";
                    return Err(RuntimeError::from(msg));
                }
                match me.to_label_filters() {
                    Ok(filters) => Ok(filters),
                    Err(err) => {
                        let msg = format!("error parsing metric selector; {:?}", err);
                        Err(RuntimeError::from(msg))
                    }
                }
            }
            _ => {
                let msg = format!("expecting metric selector; got {}", expr);
                Err(RuntimeError::from(msg))
            }
        },
        Err(err) => Err(RuntimeError::ParseError(err)),
    };
}
