use std::fmt::Debug;
use std::sync::Arc;

use super::MetricName;
use crate::runtime_error::{RuntimeError, RuntimeResult};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Timeseries {
    pub metric_name: MetricName,
    pub values: Vec<f64>,
    pub timestamps: Arc<Vec<i64>>, //Arc used vs Rc since Rc is !Send
}

impl Timeseries {
    pub fn new(timestamps: Vec<i64>, values: Vec<f64>) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values,
            timestamps: Arc::new(timestamps),
        }
    }

    pub fn copy(src: &Timeseries) -> Self {
        Timeseries {
            timestamps: src.timestamps.clone(),
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
        }
    }

    pub fn copy_from_metric_name(src: &Timeseries) -> Self {
        let ts = Timeseries {
            timestamps: Arc::clone(&src.timestamps),
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
        };
        ts
    }

    pub fn with_shared_timestamps(timestamps: &Arc<Vec<i64>>, values: &[f64]) -> Self {
        Timeseries {
            metric_name: MetricName::default(),
            values: Vec::from(values),
            // see https://pkolaczk.github.io/server-slower-than-a-laptop/ under the section #the fix
            timestamps: Arc::new(timestamps.as_ref().clone()), // clones the value under Arc and wraps it in a new counter
        }
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.timestamps = Arc::new(vec![]);
        self.metric_name.reset();
    }

    pub fn copy_from(src: &Timeseries) -> Self {
        Timeseries {
            metric_name: src.metric_name.clone(),
            values: src.values.clone(),
            timestamps: Arc::clone(&src.timestamps),
        }
    }

    pub fn is_all_nans(&self) -> bool {
        self.values.iter().all(|v| v.is_nan())
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn tag_count(&self) -> usize {
        self.metric_name.tags.len()
    }
}

pub(crate) fn assert_identical_timestamps(tss: &[Timeseries], step: i64) -> RuntimeResult<()> {
    if tss.len() == 0 {
        return Ok(());
    }
    let ts_golden = &tss[0];
    if ts_golden.values.len() != ts_golden.timestamps.len() {
        let msg = format!(
            "BUG: ts_golden.values.len() must match ts_golden.timestamps.len(); got {} vs {}",
            ts_golden.values.len(),
            ts_golden.timestamps.len()
        );
        return Err(RuntimeError::from(msg));
    }
    if ts_golden.timestamps.len() > 0 {
        let mut prev_timestamp = ts_golden.timestamps[0];
        for timestamp in ts_golden.timestamps.iter().skip(1) {
            if timestamp - prev_timestamp != step {
                let msg = format!(
                    "BUG: invalid step between timestamps; got {}; want {};",
                    timestamp - prev_timestamp,
                    step
                );
                return Err(RuntimeError::from(msg));
            }
            prev_timestamp = *timestamp
        }
    }
    for ts in tss.iter() {
        if ts.values.len() != ts_golden.values.len() {
            let msg = format!(
                "BUG: unexpected ts.values.len(); got {}; want {}; ts.values={}",
                ts.values.len(),
                ts_golden.values.len(),
                ts.values.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() != ts_golden.timestamps.len() {
            let msg = format!(
                "BUG: unexpected ts.timestamps.len(); got {}; want {};",
                ts.timestamps.len(),
                ts_golden.timestamps.len()
            );
            return Err(RuntimeError::from(msg));
        }
        if ts.timestamps.len() == 0 {
            continue;
        }
        if &ts.timestamps[0] == &ts_golden.timestamps[0] {
            // Fast path - shared timestamps.
            continue;
        }
        for (ts, golden) in ts.timestamps.iter().zip(ts_golden.timestamps.iter()) {
            if ts != golden {
                let msg = format!("BUG: timestamps mismatch; got {}; want {};", ts, golden);
                return Err(RuntimeError::from(msg));
            }
        }
    }
    Ok(())
}

pub(crate) fn get_timeseries() -> Timeseries {
    // timeseries_pool().pull().timeseries.borrow_mut()
    Timeseries::default()
}
