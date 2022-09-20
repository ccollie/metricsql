use std::borrow::Cow;
use std::fmt;
use std::fmt::Display;
use std::ops::Add;

use chrono::Duration;

use metricsql::ast::{Expression, LabelFilter};
use metricsql::parser::parse;

use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::traits::{Timestamp, TimestampTrait};

pub(crate) fn round_to_seconds(ms: i64) -> i64 {
     ms - &ms % 1000
}

/// These values prevent from overflow when storing msec-precision time in int64.
const MIN_TIME_MSECS: i64  = 0; // use 0 instead of `int64(-1<<63) / 1e6` because the storage engine doesn't actually support negative time
const MAX_TIME_MSECS: i64 = ((1 << 63 - 1) as f64 / 1e6) as i64;
const MAX_DURATION_MSECS: i64 = 100 * 365 * 24 * 3600 * 1000;


/// Deadline contains deadline with the corresponding timeout for pretty error messages.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Deadline {
    /// deadline in unix timestamp seconds.
    pub deadline: Timestamp,
    pub timeout: Duration
}

impl Deadline {
    /// Returns a deadline for the given timeout.
    pub fn new(timeout: Duration) -> Self {
        Deadline::new_ex(Timestamp::now(), timeout)
    }

    /// Returns a deadline for the given timeout.
    pub fn new_ex(start_time: Timestamp, timeout: Duration) -> Self {
        return Deadline {
            deadline: start_time.add(timeout.num_milliseconds()),
            timeout,
        }
    }


    /// returns true if deadline is exceeded.
    pub fn exceeded(&self) -> bool {
        Timestamp::now() > self.deadline
    }
}

impl Default for Deadline {
    fn default() -> Self {
        Self::new(Duration::milliseconds(0))
    }
}

impl Display for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start_time = self.deadline  - self.timeout.num_milliseconds();
        let elapsed = (Timestamp::now() - start_time) / 1000_i64;
        write!(f, "{:.3} seconds (elapsed {:.3} seconds);",
                           self.timeout.num_seconds(), elapsed)?;
        Ok(())
    }
}


/// join_tag_filterss adds etfs to every src filter and returns the result.
pub(crate) fn join_tag_filterss<'a>(src: &Vec<Vec<LabelFilter>>, etfs: &Vec<Vec<LabelFilter>>) -> Cow<'a, Vec<Vec<LabelFilter>>> {
    if src.len() == 0 {
        return Cow::Borrowed(etfs)
    }
    if etfs.len() == 0 {
        return Cow::Borrowed(src)
    }
    let mut dst: Vec<Vec<LabelFilter>> = Vec::with_capacity(src.len());
    for tf in src {
        let mut tfs: Vec<LabelFilter> = tf.clone();
        for etf in etfs {
            tfs.append( &mut etf.clone());
            dst.push(tfs.into());
        }
    }
    Cow::Owned(dst)
}

/// parse_metric_selector parses s containing PromQL metric selector and returns the corresponding 
/// LabelFilters.
pub fn parse_metric_selector(s: &str) -> RuntimeResult<Vec<LabelFilter>> {
    return match parse(s) {
        Ok(expr) => {
            match expr {
                Expression::MetricExpression(me) => {
                    if me.label_filters.len() == 0 {
                        let msg = "labelFilters cannot be empty";
                        return Err(RuntimeError::from(msg));
                    }
                    Ok(me.label_filters.into())
                },
                _ => {
                    let msg = format!("expecting metric selector; got {}", expr);
                    Err(RuntimeError::from(msg))
                }
            }
        },
        Err(err) => {
            Err(
                RuntimeError::ParseError( err )
            )
        }
    }
}
