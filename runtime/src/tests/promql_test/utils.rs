// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc};
use crate::{QueryValue, RuntimeResult, Timeseries};
use crate::tests::promql_test::types::TestAssertionError;
use super::parser::TEST_START_TIME;

// Constants
pub const DEFAULT_EPSILON: f64 = 0.000001; // Relative error allowed for sample values.

pub fn almost_equal(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

pub fn timestamp_from_datetime(d: &DateTime<Utc>) -> i64 {
    d.timestamp() * 1000 + d.timestamp_subsec_millis() as i64
}

pub fn timestamp_from_system_time(d: &SystemTime) -> i64 {
    d.duration_since(TEST_START_TIME).unwrap().as_millis() as i64
}

pub fn unix_millis_to_system_time(unix_millis: i64) -> SystemTime {
    let duration = Duration::from_millis(unix_millis as u64);
    UNIX_EPOCH + duration
}

pub(super) fn format_series_result(s: &Timeseries) -> String {
    let mut float_plural = "s";

    if s.values.len() == 1 {
        float_plural = "";
    }

    format!("{} float point{} {}",
            s.len(), float_plural, s.values.iter().map(|v| format!("{:.3}", v)).collect::<Vec<String>>().join(", "))
}

pub fn assert_matrix_sorted(m: &QueryValue) -> Result<(), TestAssertionError> {
    match m {
        QueryValue::RangeVector(m) => {
            if m.is_empty() {
                return Ok(());
            }

            for i in 0..m.len() - 1 {
                let next_index = i + 1;
                let next_metric = &m[next_index].metric_name;

                if m[i].metric_name > *next_metric {
                    let msg = format!("matrix results should always be sorted by labels, but matrix is not sorted: series at index {} with labels {} sorts before series at index {} with labels {}",
                                      next_index, next_metric, i, m[i].metric_name);
                    return Err(TestAssertionError::new(0, msg));
                }
            }
        }
        _ => {}
    }
    Ok(())
}
