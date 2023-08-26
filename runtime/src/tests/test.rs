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

// https://github.com/prometheus/prometheus/blob/main/promql/test.go

use std::cell::OnceCell;
use std::fmt::Display;
use std::fs;
use std::ops::Index;
use std::time::Duration;

use chrono::{DateTime, Utc};
use regex::Regex;

use metricsql::ast::DurationExpr;
use metricsql::parser::ParseErr;
use metricsql::utils::parse_number;

use crate::{Context, MetricName, RuntimeError, RuntimeResult, Timestamp};
use crate::tests::clear_cmd::ClearCmd;
use crate::tests::consts::{eval_instant_regex, load_regex, space_regex};
use crate::tests::eval_cmd::EvalCmd;
use crate::tests::load_cmd::LoadCmd;
use crate::tests::test_storage::TestStorage;
use crate::tests::types::{CancelFunc, SequenceValue};

pub struct TestParseError {
    pub line_offset: usize,
    pub err: String,
    pub span: Span,
}

fn sample_regex() -> &'static Regex {
    static SAMPLE_RE: OnceCell<Regex> = OnceCell::new();
    SAMPLE_RE.get_or_init(|| {
        Regex::new(r"^(?P<name>\w+)(\{(?P<labels>[^}]+)})?\s+(?P<value>\S+)(\s+(?P<timestamp>\S+))?")
            .unwrap()
    })
}

/// AtModifierUnsafeFunctions are the functions whose result
/// can vary if evaluation time is changed when the arguments are
/// step invariant. It also includes functions that use the timestamps
/// of the passed instant vector argument to calculate a result since
/// that can also change with change in eval time.
fn is_at_modifier_unsafe_functions() -> bool {
    todo!()
}

// Step invariant functions.
// "days_in_month", "day_of_month", "day_of_week", "day_of_year",
// "hour", "minute", "month", "year",
// "predict_linear", "time",
// // Uses timestamp of the argument for the result,
// // hence unsafe to use with @ modifier.
// "timestamp"
// }

static TEST_START_TIME: OnceCell<DateTime<Utc>> = OnceCell::new();

pub(crate) fn test_start_time() -> &'static DateTime<Utc> {
    TEST_START_TIME.get_or_init(|| Utc::now())
}


pub struct SeriesDescription {
    labels: MetricName,
    values: Vec<SequenceValue>,
}

pub enum TestCommand {
    Load(LoadCmd),
    Clear(ClearCmd),
    Eval(EvalCmd),
}

impl TestCommand {
    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        match self {
            TestCommand::Clear(cmd) => cmd.exec(t),
            TestCommand::Load(cmd) => cmd.exec(t),
            TestCommand::Eval(cmd) => cmd.exec(t)
        }
    }

    pub fn is_fail(&self) -> bool {
        match self {
            TestCommand::Clear(cmd) => cmd.fail,
            TestCommand::Load(cmd) => cmd.fail,
            TestCommand::Eval(cmd) => cmd.fail
        }
    }
}

/// Test is a sequence of read and write commands that are run
/// against a test storage.
pub(crate) struct Test {
    cmds: Vec<TestCommand>,
    pub(crate) storage: TestStorage,
    query_engine: Engine,
    context: Context,
    cancel_ctx: Option<CancelFunc>,
}

impl Test {
    // returns an initialized empty Test.
    pub fn new(input: &str) -> RuntimeResult<Test> {
        let mut test = Test {
            cmds: vec![],
            storage: TestStorage::new(),
            query_engine: (),
            context: Context::default(),
            cancel_ctx: None,
        };
        test.parse(input)?;
        test.clear();

        Ok(test)
    }

    pub fn from_file(filename: &str) -> RuntimeResult<Test> {
        let content = fs::read(filename)?;
        return Self::new(content.as_str());
    }

    /// parse the given command sequence and appends it to the test.
    fn parse(&mut self, input: &str) -> RuntimeResult<()> {
        let lines = get_lines(input);
        // Scan for steps line by line.
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let parts = space_regex().split(line).collect();
            let cmd_str = parts.get(2).unwrap_or("".to_string()).to_lower();

            let cmd = match cmd_str {
                "clear" => TestCommand::Clear(ClearCmd::new()),
                "load" => {
                    let (i, cmd) = parse_load(&lines, i)?;
                    cmd
                }
                _ => {
                    if cmd_str.starts_with("eval") {
                        let (i, cmd) = self.parse_eval(&lines, i)?;
                        cmd
                    } else {
                        return Err(raise(i, format!("invalid command {cmd}").as_str()));
                    }
                }
            };

            self.cmds.push(cmd)
        }

        Ok(())
    }

    // clear the current test storage of all inserted samples.
    pub fn clear(&mut self) {
        self.storage.close();
        if let Some(cancel_ctx) = &self.cancel_ctx {
            (cancel_ctx)()
        }
        self.storage = TestStorage::new();
        let opts = EngineOpts {
            max_samples: 10000,
            timeout: Duration::from_secs(100),
            NoStepSubqueryIntervalFn: Duration::from_millis(1000 * 60),
        };

        self.queryEngine = NewEngine(opts)
    }

    /// Close closes resources associated with the Test.
    pub fn close(&mut self) -> RuntimeResult<()> {
        if let Some(cancel_ctx) = &self.cancel_ctx {
            (cancel_ctx)()
        }
        self.storage.close()
            .map_err(|e| RuntimeError::from("Unexpected error while closing test storage."))
    }

    /// Run executes the command sequence of the test. Until the maximum error number
    /// is reached, evaluation errors do not terminate execution.
    pub fn run(&mut self) -> RuntimeResult<()> {
        for cmd in self.cmds.iter_mut() {
            // TODO(fabxc): aggregate command errors, yield diffs for result
            // comparison errors.
            cmd.exec(self)?;
        }
        Ok(())
    }

    fn parse_eval(&mut self, lines: &Vec<&String>, i: usize) -> RuntimeResult<(usize, EvalCmd)> {
        let line = &lines[i];

        let (mod_, at, expr) = if let Some(captures) = eval_instant_regex().captures(&line) {
            let mod_ = captures.get(1).unwrap().as_str();
            let at = captures.get(2).unwrap().as_str();
            let expr = captures.get(3).unwrap().as_str();
            (mod_, at, expr)
        } else {
            return Err(
                raise(i, "invalid evaluation command. (eval[_fail|_ordered] instant [at <offset:duration>] <query>")
            );
        };

        match metricsql::parser::parse(expr) {
            Err(err) => {
                // Adjust error position to line and column.
                // (parser errors are 0-indexed
                let mut perr: ParseErr;
                if errors.As(err, &perr) {
                    perr.line_offset = i;
                    let pos_offset = line.index(expr)?;
                    perr.span.start += pos_offset;
                    perr.span.end += pos_offset;
                    perr.query = line.to_string()
                }
                return Err(err);
            }
            Ok(_) => {}
        }

        let offset = parse_duration(at)
            .map_err(|err| {
                RuntimeError::General(format!("invalid step definition {}: {:?}", at, err));
            })?;

        let ts = test_start_time.add(Duration::from_millis(offset as u64));
        let mut cmd = EvalCmd::new(expr, ts, i + 1);
        match mod_ {
            "ordered" => cmd.ordered = true,
            "fail" => cmd.fail = true,
            _ => {}
        }

        let mut i = i;
        let mut j = 1;
        while i + 1 < lines.len() {
            i += 1;
            let def_line = &lines[i];
            if def_line.len() == 0 {
                i -= 1;
                break;
            }
            if let Some(f) = parse_number(&def_line) {
                self.expect(0, SequenceValue { value: f, omitted: false });
                break;
            }
            let (metric, vals) = parse_series_desc(def_line);
            if err != nil {
                let perr: ParseErr;
                if errors.As(err, &perr) {
                    perr.line_offset = i
                }
                return Err(err);
            }

            // Currently, we are not expecting any matrices.
            if vals.len() > 1 {
                return Err(raise(i, "expecting multiple values in instant evaluation not allowed"));
            }
            cmd.expect(j, metric, vals)
        }
        Ok((i, cmd))
    }
}


fn raise(line: usize, err: &str) -> RuntimeError {
    return RuntimeError::ParseErr {
        line_offset: line,
        err,
    };
}

pub(super) fn parse_load(lines: &[String], i: usize) -> RuntimeResult<(usize, LoadCmd)> {
    let line = &lines[i];
    let gap = if let Some(captures) = load_regex().captures(&line) {
        captures.get(1).unwrap().as_str()
    } else {
        return Err(raise(i, "invalid load command. (load <step:duration>)"));
    };

    let gap = parse_duration(gap)
        .map_err(|err| {
            RuntimeError::General(format!("invalid step definition {}: {:?}", gap, err));
        })?;

    let mut cmd = LoadCmd::new(Duration::from_millis(gap as u64));
    let mut i = i;
    while i + 1 < lines.len() {
        i += 1;
        if lines[i].is_empty() {
            i -= 1;
            break;
        };
        let (metric, vals) = parse_series_desc(&lines[i]);
        if err != nil {
            let perr: ParseErr;
            if errors.As(err, &perr) {
                perr.line_offset = i
            }
            return Err(err);
        }
        cmd.set(metric, vals)
    }
    return Ok((i, cmd));
}

/// get_lines returns trimmed lines after removing the comments.
pub(crate) fn get_lines(input: &str) -> Vec<String> {
    let lines = input.split("\n");
    let mut result = Vec::new();
    for l in lines.iter() {
        let mut l = l.trim();
        if l.starts_with("#") {
            l = ""
        }
        result.push(l.clone());
    }
    result
}

pub(crate) struct AtModifierTestCase {
    pub(crate) expr: String,
    pub(crate) eval_time: Timestamp,
}

pub(crate) fn at_modifier_test_cases(expr_str: String, eval_time: Timestamp) -> RuntimeResult<Vec<AtModifierTestCase>> {
    let expr = metricsql::parser::parse(&expr_str)?;
    let ts = timestamp.FromTime(eval_time);

    let contains_non_step_invariant = false;
    // Setting the @ timestamp for all selectors to be eval_time.
    // If there is a subquery, then the selectors inside it don't get the @ timestamp.
    // If any selector already has the @ timestamp set, then it is untouched.
    parser.inspect(expr, |node: &mut Expr, path: &[Node]| -> RuntimeResult<()> {
        let mut subq_ts = subquery_times(path);
        match node {
            Expr::MetricExpression(me) => {
                if n.timestamp == nil {
                    n.timestamp = makeInt64Pointer(ts)
                }

                Expr::MatrixSelector(me) => {
                    if vs = n.VectorSelector.(*parser.VectorSelector);
                    vs.timestamp == 0
                    {
                        vs.timestamp = makeInt64Pointer(ts)
                    }
                }

                Expr::RollupExpr(re) => {
                    if n.timestamp == nil {
                        n.timestamp = makeInt64Pointer(ts)
                    }
                }
            }
            Expr::Function(fe) => {
                let ok = AtModifierUnsafeFunctions[n.func.name];
                contains_non_step_invariant = contains_non_step_invariant || ok;
            }
        })

        if contains_non_step_invariant {
            // Since there is a step invariant function, we cannot automatically
            // generate step invariant test cases for it sanely.
            return nil;
        }

        let new_expr = expr.to_string(); // With all the @ eval_time set.
        let mut additional_eval_times = &[-10 * ts, 0, ts / 5, ts, 10 * ts];
        if ts == 0 {
            additional_eval_times = &[-1000, -ts, 1000];
        }
        let mut test_cases = Vec::with_capacity(additional_eval_times.len());
        for et in additional_eval_times.iter() {
            test_cases.push(AtModifierTestCase {
                expr: new_expr,
                eval_time: timestamp.Time(et),
            })
        }
        Ok(test_cases)
    }
}

// ParseSeriesDesc parses the description of a time series.
fn parse_series_desc(input: &str) -> RuntimeResult<SeriesDescription> {
    let p = newParser(input);

    let parse_result = p.parse_generated(START_SERIES_DESCRIPTION)?;
    let result = parse_result.(*SeriesDescription)
    labels = result.labels
    values = result.values

    return SeriesDescription { labels, values };
}

fn parse_duration(input: &str) -> RuntimeResult<i64> {
    let d = DurationExpr::try_from(input).map_err(|e| {
        RuntimeError::General(format!("invalid duration definition {}: {:?}", input, e));
    });
    d.value(1)
}

// subquery_times returns the sum of offsets and ranges of all subqueries in the path.
// If the @ modifier is used, then the offset and range is w.r.t. that timestamp
// (i.e. the sum is reset when we have @ modifier).
// The returned *int64 is the closest timestamp that was seen. nil for no @ modifier.
fn subquery_times(path: &[&Expr]) -> (Duration, Duration, i64) {
    let mut subq_offset = Duration::from_secs(0);
    let mut subq_range = Duration::from_secs(0);
    let mut ts = i64::MAX;

    for node in path.iter() {
        if matches!(node, Expr::Rollup(n)) {
            subq_offset += n.OriginalOffset;
            subq_range += n.Range;
            if n.Timestamp != nil {
                // The @ modifier on subquery invalidates all the offset and
                // range till now. Hence resetting it here.
                subq_offset = n.OriginalOffset;
                subq_range = n.Range;
                ts = *n.Timestamp
            }
        }
    }
    var
    tsp * int64
    if ts != math.MaxInt64 {
        tsp = &ts
    }
    return subq_offset, subq_range, tsp
}