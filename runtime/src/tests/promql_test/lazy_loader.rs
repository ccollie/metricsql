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
use super::parser::parse_load;
use super::test_command::{LoadCmd, TestCommand};
use crate::execution::Context;
use crate::{MemoryMetricProvider, MetricStorage};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

// LazyLoader lazily loads samples into storage.
// This is specifically implemented for unit testing of rules.
pub struct LazyLoader {
    load_cmd: Option<LoadCmd>,
    storage: Arc<MemoryMetricProvider>,
    subquery_interval: Duration,
    context: Context,
    options: PromqlEngineOpts,
}

impl LazyLoader {
    // NewLazyLoader returns an initialized empty LazyLoader.
    pub fn new(input: &str) -> Result<Self, Box<dyn Error>> {
        let storage = Arc::new(MemoryMetricProvider::default());
        let context: Context = Context::new().with_metric_storage(storage.clone());
        let mut ll = LazyLoader {
            load_cmd: None,
            context,
            storage,
            subquery_interval: Duration::from_secs(0),
            options: PromqlEngineOpts::default(),
        };
        ll.parse(input)?;
        ll.clear()?;
        Ok(ll)
    }

    // parse the given load command.
    pub fn parse(&mut self, input: &str) -> Result<(), Box<dyn Error>> {
        let lines = get_lines(input);
        let mut i: usize = 0;
        // Accepts only 'load' command.
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() {
                i += 1;
                continue;
            }
            if line.to_lowercase().starts_with("load") {
                let (j, cmd) = parse_load(&lines, i)?;
                i = j;
                return match cmd {
                    TestCommand::Load(cmd) => {
                        self.load_cmd = Some(cmd);
                        Ok(())
                    }
                    _ => Err(Box::new(InvalidCommandError(i, line.to_string()))),
                }
            }
            return Err(Box::new(InvalidCommandError(i, line.to_string())));
        }
        Err(Box::new(NoLoadCommandError))
    }

    // clear the current test storage of all inserted samples.
    fn clear(&mut self) -> Result<(), Box<dyn Error>> {
        self.storage.clear();
        self.options = PromqlEngineOpts {
            max_samples: 10000,
            timeout: Duration::from_secs(100),
            no_step_subquery_interval_fn: Box::new(|_| self.subquery_interval.as_millis() as i64),
            enable_delayed_name_removal: true,
        };
        Ok(())
    }

    // appendTill appends the defined time series to the storage till the given timestamp (in milliseconds).
    fn append_till(&mut self, ts: i64) -> Result<(), Box<dyn Error>> {
        if let Some(mut load_cmd) = &self.load_cmd {
            for (h, smpls) in &load_cmd.defs {
                if let Some(m) = load_cmd.metrics.get(h) {
                    for (i, s) in smpls.iter().enumerate() {
                        if s.timestamp > ts {
                            // Removing the already added samples.
                            load_cmd.defs.get_mut(h).unwrap().drain(..i);
                            break;
                        }
                        self.storage.append(m.clone(), s.timestamp, s.value)?;
                        if i == smpls.len() - 1 {
                            load_cmd.defs.get_mut(h).unwrap().clear();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // WithSamplesTill loads the samples till given timestamp and executes the given function.
    fn with_samples_till<F>(&mut self, ts: Duration, mut fn_: F)
    where
        F: FnMut(Result<(), Box<dyn Error>>),
    {
        let ts_milli = ts.as_millis() as i64;
        fn_(self.append_till(ts_milli));
    }

    // Queryable allows querying the LazyLoader's data.
    // Note: only the samples till the max timestamp used
    // in `WithSamplesTill` can be queried.
    fn queryable(&self) -> Arc<dyn MetricStorage> {
        self.storage.clone()
    }

    // Storage returns the LazyLoader's storage.
    fn storage(&self) -> Arc<dyn MetricStorage> {
        self.storage.clone()
    }
}

// Custom error types for better error handling.
#[derive(Debug)]
struct InvalidCommandError(usize, String);

impl fmt::Display for InvalidCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid command at line {}: {}", self.0, self.1)
    }
}

impl Error for InvalidCommandError {}

#[derive(Debug)]
struct NoLoadCommandError;

impl fmt::Display for NoLoadCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no \"load\" command found")
    }
}

impl Error for NoLoadCommandError {}

pub struct PromqlEngineOpts {
    max_samples: usize,
    timeout: Duration,
    no_step_subquery_interval_fn: Box<dyn Fn(i64) -> i64>,
    enable_delayed_name_removal: bool,
}

impl Default for PromqlEngineOpts {
    fn default() -> Self {
        PromqlEngineOpts {
            max_samples: 10000,
            timeout: Duration::from_secs(100),
            no_step_subquery_interval_fn: Box::new(|_| 0),
            enable_delayed_name_removal: false,
        }
    }
}


// Placeholder functions for the actual implementations.
fn get_lines(input: &str) -> Vec<String> {
    input.lines().map(|s| s.to_string()).collect()
}