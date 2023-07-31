use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};

use crate::MetricName;

/// Point represents a single data point for a given timestamp.
/// If H is not nil, then this is a histogram point and only (T, H) is valid.
/// If H is nil, then only (T, V) is valid.
#[derive(Debug, Default)]
struct Point {
    t: i64,
    v: f64,
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} @[{}]", self.v, self.t)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    metric: MetricName,
    t: i64,
    v: f64,
}

impl Sample {
    pub fn new(labels: MetricName, t: i64, v: f64) -> Self {
        Self {
            metric: labels,
            t,
            v,
        }
    }

    pub fn from_hashmap(map: &HashMap<String, String>, t: i64, v: f64) -> Self {
        let mut metric_name = MetricName::new("");
        for (k, v) in map.iter() {
            metric_name.set_tag(k.as_str(), v)
        }
        Self {
            metric: metric_name,
            t,
            v,
        }
    }
}

/// SequenceValue is an omittable value in a sequence of time series values.
pub(crate) struct SequenceValue {
    pub value: f64,
    pub omitted: bool,
}

impl Display for SequenceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.omitted {
            write!(f, "_")
        }
        write!(f, "{}", self.value)
    }
}

pub(crate) type CancelFunc = fn() -> ();
