use super::{MetricName, Timestamp};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Sample {
    metric: MetricName,
    timestamp: Timestamp,
    value: f64
}
