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
use std::fs;
use crate::execution::{exec_internal, Context, EvalConfig};
use super::parser::{parse_eval, parse_expr, parse_load};
use super::test_command::{ClearCmd, EvalCmd, TestCommand};
use super::utils::{
    assert_matrix_sorted,
    timestamp_from_datetime,
    timestamp_from_system_time,
    unix_millis_to_system_time
};
use super::types::{ParseErr, Sample};
use crate::{MemoryMetricProvider, QueryValue, RuntimeResult};
use metricsql_parser::ast::Expr;
use metricsql_parser::ast::Expr::{Aggregation, BinaryOperator, Function, Rollup};
use regex::Regex;
use std::sync::Arc;
use std::time::SystemTime;
use glob::glob;

const ONE_MINUTE_AS_MILLIS: i64 = 60 * 1000;

static PAT_SPACE: Regex = Regex::new(r"[\t ]+").unwrap();

// LoadedStorage returns storage with generated data using the provided load statements.
// Non-load statements will cause test errors.
pub fn loaded_storage(input: &str) -> Arc<MemoryMetricProvider> {
    let mut test = Test::new(input);

    for cmd in test.cmds.iter_mut() {
        match cmd {
            TestCommand::Load(cmd) => {
                cmd.append(&test.storage);
            }
            _ => {
                panic!("only 'load' commands accepted, got '{:?}'", cmd);
            }
        }
    }
    test.storage
}

// run_builtin_tests runs an acceptance test suite against the provided engine.
pub fn run_builtin_tests() {
    for entry in glob("**/*.test").expect("Failed to read glob pattern") {
        match entry.as_ref() {
            Ok(path) => {
                match fs::read_to_string(path) {
                    Ok(content) => {
                        run_test(&content).unwrap();
                    }
                    Err(e) => {
                        println!("Error loading test file {:?}: {:?}", path, e);
                    }
                }
            },
            Err(e) => println!("Error: {:?}", e),
        }
    }
}

pub fn run_test(input: &str) -> Result<(), String> {
    let mut test = Test::new(input);
    for cmd in test.cmds.iter_mut() {
        test.exec(cmd)?;
        // TODO(fabxc): aggregate command errors, yield diffs for result
    }
    Ok(())
}

// test is a sequence of read and write commands that are run
// against a test storage.
pub struct Test {
    pub(super) cmds: Vec<TestCommand>,
    storage: Arc<MemoryMetricProvider>,
    context: Context
}


impl Test {
    pub fn new(input: &str) -> Test {
        let storage = Arc::new(MemoryMetricProvider::default());
        let context = Context::new().with_metric_storage(storage.clone());
        let mut test = Test{
            cmds: vec![],
            storage,
            context,
        };
        test.parse(input).unwrap();
        test.clear();

        test
    }

    // parse the given command sequence and appends it to the test.
    pub fn parse(&mut self, input: &str) -> Result<(), ParseErr> {
        let mut lines = get_lines(input);
        // Scan for steps line by line.
        let mut i: usize = 0;
        while i < lines.len() {
            let line = &lines[i];
            if line.is_empty() {
                i += 1;
                continue;
            }
            let c = PAT_SPACE.split(line).next().unwrap_or_default().to_lower();
            let cmd = match c {
                "clear" => TestCommand::Clear(ClearCmd{}),
                _ if c.starts_with("load") => {
                    let (j, cmd) = parse_load(line, i)?;
                    i = j;
                    cmd
                },
                _ if c.starts_with("eval") => {
                    let (j, cmd) = parse_eval(line, i);
                    i = j;
                    cmd
                },
                _ => return Err(ParseErr {
                    line_offset: i,
                    position_range: (0, 0),
                    query: line.clone(),
                    err: format!("unknown command: {}", line),
                }.into())
            };
            self.cmds.push(cmd);
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.storage.clear();
    }

    // exec processes a single step of the test.
    pub fn exec(&mut self, tc: &mut TestCommand) -> Result<(), String> {
        match tc {
            TestCommand::Clear(_) => self.clear(),
            TestCommand::Load(cmd) => {
                let storage = self.storage.clone();
                cmd.append(&storage);
            }
            TestCommand::Eval(cmd) => self.execEval(cmd)
        }
        Ok(())
    }

    fn exec_eval(&mut self, cmd: &EvalCmd) -> Result<(), String> {
        if cmd.is_range {
            return self.exec_range_eval(cmd)
        }

        self.exec_instant_eval(cmd)
    }

    fn exec_instant_eval(&mut self, cmd: &EvalCmd) -> Result<(), String> {
        let mut queries = at_modifier_test_cases(&cmd.expr, &cmd.start);
        queries.insert(0, AtModifierTestCase{
            expr: cmd.expr.clone(),
            eval_time: cmd.start
        });
        for iq in queries.iter() {
            self.run_instant_query(iq, cmd)?;
        }
        Ok(())
    }

    fn run_instant_query(&mut self, iq: &AtModifierTestCase, cmd: &EvalCmd) -> Result<(), String> {
        let mut ec = new_instant_query(&iq.eval_time)
            .map_err(|err| {
                format!("error creating instant query for {} (line {}): {:?}", cmd.expr, cmd.line, err)
            })?;

        let res = self.exec_internal(&mut ec, &cmd.expr);
        if let Err(e) = res {
            if cmd.fail {
                cmd.check_expected_failure(e)?;
            }
            return Err(format!("error evaluating query {} (line {}): {:?}", iq.expr, cmd.line, e))
        } else if cmd.fail {
            return Err(format!("expected error evaluating query {} (line {}) but got none", iq.expr, cmd.line))
        }
        let res = res?;
        let eval_time = timestamp_from_datetime(iq.eval_time);
        cmd.compare_result(res)
            .map_err(|err| format!("error in {} {} (line {}): {:?}", cmd, iq.expr, cmd.line, err))?;

        // Check query returns same result in range mode,
        // by checking against the middle step.
        let start = eval_time - ONE_MINUTE_AS_MILLIS;
        let end = eval_time + ONE_MINUTE_AS_MILLIS;
        let mut ec = new_range_query(start, end, ONE_MINUTE_AS_MILLIS)
            .map_err(|err| format!("error creating range query for {} (line {}): {:?}", cmd.expr, cmd.line, err))?;

        let range_res = self.exec_internal(&mut ec, &cmd.expr)
            .map_err(|err| format!("error evaluating query {} (line {}) in range mode: {:?}", iq.expr, cmd.line, err))?;

        if cmd.ordered {
            // Range queries are always sorted by labels, so skip this test case that expects results in a particular order.
            return Ok(())
        }
        match range_res {
            QueryValue::Scalar(v) => {
                cmd.compare_result(&range_res)
                    .map_err(|err| format!("error in {} {} (line {}) range mode: {:?}", cmd, iq.expr, cmd.line, err))?;
            }
            QueryValue::RangeVector(mat) => {
                assert_matrix_sorted(mat)?;

                let mut vec = Vec::with_capacity(mat.len());
                for series in mat.iter() {
                    for (timestamp, value) in series.timestamps.iter().zip(series.values.iter()) {
                        if timestamp == eval_time {
                            vec.push(Sample{
                                timestamp,
                                value,
                                metric: series.metric_name.clone(),
                            });
                            break
                        }
                    }
                }
                cmd.compare_result(vec)
            }
            QueryValue::InstantVector(vec) => {
                let mut to_compare = Vec::with_capacity(vec.len());
                for series in vec.iter() {
                    let timestamp = series.timestamps[0];
                    if timestamp == eval_time {
                        let value = series.values[0];
                        to_compare.push(Sample {
                            metric: series.metric_name.clone(),
                            timestamp,
                            value
                        })
                    }
                }
                cmd.compare_result(vec)
            }
            _ => return Err(format!("unexpected query result type: {:?}", range_res))
        }

        Ok(())
}

    fn exec_range_eval(&self, cmd: &EvalCmd) -> Result<(), String> {
        let step = cmd.step.as_millis() as i64;
        let mut ec = new_range_query(&cmd.start, &cmd.end, step)
            .map_err(|err| format!("error creating range query for {} (line {}): {:?}", cmd.expr, cmd.line, err))?;
        let res = self.exec_internal(&mut ec, &cmd.expr);
        match res {
            Ok(v) => {
                if cmd.fail {
                    let msg = format!("expected error evaluating query {} (line {}) but got none", cmd.expr, cmd.line);
                    return Err(msg)
                }
            }
            Err(e) => {
                if cmd.fail {
                    cmd.checkExpectedFailure(e)?;
                }
                let msg = format!("error evaluating query {} (line {}): {:?}", cmd.expr, cmd.line, e);
                return Err(msg)
            }
        }
        let value = res?;
        cmd.compare_result(&value)
            .map_err(|err| format!("error in {} {} (line {}): {:?}", cmd, cmd.expr, cmd.line, err))?;

        Ok(())
    }

    fn exec_internal(&self, ec: &mut EvalConfig, q: &str) -> RuntimeResult<QueryValue> {
        let (qv, parsed) = exec_internal(&self.context, ec, q)?;
        Ok(qv)
    }
}

fn new_range_query(start: &SystemTime, end: &SystemTime, step: i64) -> RuntimeResult<EvalConfig> {
    let start = timestamp_from_system_time(start);
    let end = timestamp_from_system_time(&end);
    let config = EvalConfig::new(start, end, step);
    Ok(config)
}

fn new_instant_query(eval_time: &SystemTime) -> RuntimeResult<EvalConfig> {
    let eval_time = timestamp_from_system_time(&eval_time);
    let config = EvalConfig::new(eval_time, eval_time, 0);
    Ok(config)
}


pub struct AtModifierTestCase{
    expr: String,
    eval_time: SystemTime
}

fn has_at_modifier(expr: &Expr) -> bool {
    if let Rollup(re) = expr {
        if re.at.is_some() {
            return true
        }
    }
    false
}

fn at_modifier_test_cases(expr_str: &str, eval_time: &SystemTime) -> Vec<AtModifierTestCase> {
    let mut expr = parse_expr(expr_str).unwrap();
    let mut ts = timestamp_from_system_time(eval_time);

    let mut visitor = AtModifierVisitor{
        eval_time: ts,
        contains_non_step_invariant: false,
    };
    update_at_timestamp(&mut expr, &mut visitor);

    if visitor.contains_non_step_invariant {
        // Expression contains a function whose result can vary with evaluation
        // time, even though its arguments are step invariant: skip it.
        return vec![]
    }

    let new_expr = expr.to_string(); // With all the @ eval_time set.
    let mut additional_eval_times = vec![-10 * ts, 0, ts / 5, ts, 10 * ts];
    if ts == 0 {
        additional_eval_times = vec![-1000, -ts, 1000];
    }
    let mut test_cases= Vec::with_capacity(additional_eval_times.len());
    for et in additional_eval_times.iter() {
        test_cases.push(AtModifierTestCase{
            expr: new_expr.clone(),
            eval_time: unix_millis_to_system_time(*et),
        })
    }

    test_cases
}

struct AtModifierVisitor {
    contains_non_step_invariant: bool,
    eval_time: i64,
}

// Setting the @ timestamp for all selectors to be eval_time.
// If there is a subquery, then the selectors inside it don't get the @ timestamp.
// If any selector already has the @ timestamp set, then it is untouched.
fn update_at_timestamp(expr: &mut Expr, visitor: &mut AtModifierVisitor) {
    use Expr::*;

    if has_at_modifier(expr) {
        return
    }
    // todo: for bare selectors, we should create a rollup and set the @ timestamp to eval_time.
    match expr {
        Rollup(re) => {
            update_at_timestamp(&mut re.expr, visitor);
            if !re.at.is_some() {
                re.at = Some(Box::new(Expr::from(visitor.eval_time)));
            }
        }
        Function(f) => {
            let ok = is_at_modifier_unsafe_functions(f.name());
            visitor.contains_non_step_invariant = visitor.contains_non_step_invariant || ok;
            for arg in f.args.iter_mut() {
                update_at_timestamp(arg, visitor);
            }
        }
        Aggregation(agg) => {
            let ok = is_at_modifier_unsafe_functions(agg.name());
            visitor.contains_non_step_invariant = visitor.contains_non_step_invariant || ok;
            for arg in agg.args.iter_mut() {
                update_at_timestamp(arg, visitor);
            }
        }
        BinaryOperator(be) => {
            update_at_timestamp(&mut be.left, visitor);
            update_at_timestamp(&mut be.right, visitor);
        }
        _ => {}
    }
}

fn get_lines(input: &str) -> Vec<String> {
    input
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.starts_with("#"))
        .collect()
}


// AtModifierUnsafeFunctions are the functions whose result
// can vary if evaluation time is changed when the arguments are
// step invariant. It also includes functions that use the timestamps
// of the passed instant vector argument to calculate a result since
// that can also change with change in eval time.
fn is_at_modifier_unsafe_functions(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "days_in_month" => true,
        "day_of_month" => true,
        "day_of_week" => true,
        "day_of_year" => true,
        "hour" => true,
        "minute" => true,
        "month" => true,
        "year" => true,
        "time" => true,
        "timestamp" => true,
        "predict_linear" => true,
        _ => false,
    }
}