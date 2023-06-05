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

use core::num::dec2flt::parse::parse_number;
use std::collections::btree_map::BTreeMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::time::Duration;
use regex::Regex;


const PATTERN_SPACE: Lazy<Regex> = Regex::new("[\t ]+");
const PATTERN_LOAD: Lazy<Regex> = Regex::new(r"^load\s+(.+?)$");
const PATTERN_EVAL_INSTANT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^eval(?:_(fail|ordered))?\s+instant\s+(?:at\s+(.+?))?\s+(.+)$"));

static UPPER_BUCKET_RANGE: Lazy<String> = Lazy::new(|| format!("{:.3}...+Inf", UPPER_MAX));

/// Relative error allowed for sample values.
const EPSILON: f64 = 0.000001;


/// is_at_modifier_unsafe_functions are the functions whose result
/// can vary if evaluation time is changed when the arguments are
/// step invariant. It also includes functions that use the timestamps
/// of the passed instant vector argument to calculate a result since
/// that can also change with change in eval time.
fn is_at_modifier_unsafe_functions() -> bool {

}

// Step invariant functions.
"days_in_month", "day_of_month", "day_of_week", "day_of_year",
"hour", "minute", "month", "year",
"predict_linear", "time",
// Uses timestamp of the argument for the result,
// hence unsafe to use with @ modifier.
"timestamp"
}


/// Point represents a single data point for a given timestamp.
#[derive(Debug, Default, Clone, PartialEq, PartialOrd, Eq, Ord)]
struct Point {
    t: i64,
    v: f64
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!("{} @[{}]", self.v, self.t);
        Ok(())
    }
}


let testStartTime = time.Unix(0, 0).UTC()

pub enum TestCommand {
    Load(LoadCmd),
    Clear(ClearCmd),
    Eval(EvalCmd)
}

impl TestCommand {
    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        match self {
            TestCommand::Clear(cmd) => cmd.exec(t),
            TestCommand::Load(cmd) => cmd.exec(t),
            TestCommand::Eval(cmd) => cmd.exec(t)
        }
    }
}

/// Test is a sequence of read and write commands that are run
/// against a test storage.
pub(crate) struct Test {
    cmds: Vec<TestCommand>,
    storage: TestStorage,
    query_engine: Engine,
    context: Context,
    cancel_ctx: CancelFunc
}

impl Test {

    // returns an initialized empty Test.
    pub fn new(input: String) -> RuntimeResult<Test> {
        let test = Test{
            cmds: vec![],
            storage: TestStorage::new(),
            query_engine: (),
            context: (),
            cancel_ctx: ()
        };
        test.parse(input)?;
        test.clear();

        Ok(test)
    }

    pub fn from_file(filename: String) -> RuntimeResult<Test> {
        let content = fs::read(filename)?;
        return Self::new(content)
    }

    /// parse the given command sequence and appends it to the test.
    fn parse(&mut self, input: &str) -> RuntimeResult<()> {
        let lines = get_lines(input);
        // Scan for steps line by line.
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let cmd_str = Strings::toLower(PATTERN_SPACE.split(l, 2)[0]);
            let cmd: TestCommand;

            match cmd_str {
                "clear" => cmd = TestCommand::Clear(ClearCmd::new()),
                "load" => {
                    (i, cmd) = parse_load(lines, i)?
                },
                _ => {
                    if cmd_str.starts_with("eval") {
                        (i, cmd) = parse_eval(lines, i, 0)?;
                    } else {
                        return raise(i, "invalid command {}", l)
                    }
                }
            }

            self.cmds.push(cmd)
        }

        Ok(())
    }

    // clear the current test storage of all inserted samples.
    pub fn clear(&mut self) {
        if self.storage != nil {
            self.storage.close();
            require.NoError(t.T, err, "Unexpected error while closing test storage.")
        }
        if let Some(cancel_ctx) = self.cancel_ctx {
            (cancel_ctx)()
        }
        self.storage = TestStorage::new(t);
        let opts = EngineOpts {
            max_samples: 10000,
            timeout: 100 * time.Second,
            NoStepSubqueryIntervalFn: Duration::from_millis(1000 * 60)
        };

        self.queryEngine = NewEngine(opts)
            (self.context, self.cancelCtx) = context.WithCancel(context.Background())
    }

    /// Close closes resources associated with the Test.
    pub fn close(&mut self) -> RuntimeResult<()> {
        self.cancel_ctx();
        match self.storage.close() {
            Ok(_) => {},
            Err(e) => {
                require.NoError(t.T, err, "Unexpected error while closing test storage.")
            }
        }
    }

    /// Run executes the command sequence of the test. Until the maximum error number
    /// is reached, evaluation errors do not terminate execution.
    pub fn run(&mut self) -> RuntimeResult<()> {
        for cmd in self.cmds {
            // TODO(fabxc): aggregate command errors, yield diffs for result
            // comparison errors.
            cmd.exec(self)?;
        }
        Ok(())
    }

    // Queryable allows querying the test data.
    pub fn queryable(&self) -> Queryable {
        return self.storage
    }

    fn parse_eval(&mut self, lines: &[String], i: usize) -> RuntimeResult<(usize, EvalCmd)> {
        if !PATTERN_EVAL_INSTANT.is_match(lines[i]) {
            return raise(i, "invalid evaluation command. (eval[_fail|_ordered] instant [at <offset:duration>] <query>")
        }
        let parts = patEvalInstant.FindStringSubmatch(lines[i]);
        let mod_  = parts[1];
        let at   = parts[2];
        let expr = parts[3];

        if let Err(err) = parser.parse(expr) {
            let perr: ParseErr;
            if errors.As(err, &perr) {
                perr.line_offset = i;
                let pos_offset = parser.pos(strings.Index(lines[i], expr));
                perr.span.start += pos_offset;
                perr.span.end += pos_offset;
                perr.query = lines[i]
            }
            return Err(err)
        }

        let offset = parse_duration(at);
        if err != nil {
            return raise(i, format!("invalid step definition {}: {:?}", parts[1], err));
        }
        let ts = testStartTime.add(Duration::from_millis(offset));
        let cmd = EvalCmd::new(expr, ts, i+1);
        match mod_ {
            "ordered" => cmd.ordered = true,
            "fail" => cmd.fail = true,
            _ => {}
        }

        let mut j = 1;
        while i+1 < lines.len() {
            i += 1;
            let def_line = lines[i];
            if def_line.len() == 0 {
                i -= 1;
                break
            }
            if let Some(f) = parse_number(def_line, false) {
                self.expect(0, nil, SequenceValue{value: f, omitted: false });
                break
            }
            let (metric, vals) = parse_series_desc(defLine);
            if err != nil {
                let perr: ParseErr;
                if errors.As(err, &perr) {
                    perr.line_offset = i
                }
                return Err(err)
            }

            // Currently, we are not expecting any matrices.
            if vals.len() > 1 {
                raise(i, "expecting multiple values in instant evaluation not allowed");
            }
            cmd.expect(j, metric, vals...)
        }
        Ok((i, cmd))
    }
}



fn raise(line: usize, err: String) -> RuntimeResult<()> {
    return Err(ParseErr {
        line_offset: line,
        err,
    })
}

fn parse_load(lines: &[String], i: usize) -> RuntimeResult<(usize, LoadCmd)> {
    if !PATTERN_LOAD.MatchString(lines[i]) {
        return raise(i, "invalid load command. (load <step:duration>)")
    }
    let parts = PATTERN_LOAD.match(lines[i]);

    let gap = match duration_value(parts[1], 1) {
        Ok(d) => Some(d),
        Err(err) => {
            return raise(i, format!("invalid step definition {}: {}", parts[1], err))
        },
    };

    let cmd = LoadCmd::new(time.Duration(gap));
    while i+1 < lines.len() {
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
            return Err(err)
        }
        cmd.set(metric, vals...)
    }
    return Ok((i, cmd))
}

/// get_lines returns trimmed lines after removing the comments.
fn get_lines(input: String) -> Vec<String> {
    let lines = input.split("\n");
    for (i, l) in lines.iter().enumerate() {
        l = strings.TrimSpace(l);
        if l.starts_with("#") {
            l = ""
        }
        lines[i] = l
    }
    return lines
}


/// SequenceValue is an omittable value in a sequence of time series values.
pub(crate) struct SequenceValue {
    pub value: f64,
    pub omitted: bool
}

impl Display for SequenceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.omitted {
            write!(f, "_")
        }
        write!(f, "{}", self.value)
    }
}


/// load_cmd is a command that loads sequences of sample values for specific
/// metrics into the storage.
struct LoadCmd {
    gap: Duration,
    metrics: BTreeMap<u64, MetricName>,
    defs: BTreeMap<u64, Point>
}

impl LoadCmd {
    pub fn new(gap: Duration) -> LoadCmd {
        return LoadCmd{
            gap,
            metrics: Default::default(),
            defs: Default::default(),
        }
    }

    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        let mut app = t.storage.get_appender(t.context);
        match self.append(app) {
            Err(e) => {
                app.rollback();
                return Err(e);
            },
            Ok(_) => {
                app.commit();
                Ok(())
            }
        }
    }

    /// set a sequence of sample values for the given metric.
    pub fn set(&mut self, m: Labels, vals: &[SequenceValue]) {
        let h = m.hash();
        let samples: Vec<Point> = Vec::with_capacity(vals.len());
        let ts = testStartTime;
        for v in vals.iter() {
            if !v.omitted {
                samples.push(Point{
                        t: ts.milliseconds(),
                        v: v.value,
                })
            }
            ts += self.gap;
        }
        self.defs.insert(h, samples);
        self.metrics.insert(h, m);
    }

    /// append the defined time series to the storage.
    fn append(&self, a: &mut Appender) -> RuntimeResult<()> {
        for (h, samples) in self.defs.iter() {
            if let Some(m) = self.metrics.get(h) {
                for s in samples.iter() {
                    a.append(0, m, s.t, s.v);
                }
            }
        }
        Ok(())
    }
}

impl Display for LoadCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "load {:?}", self.gap)
    }
}


/// EvalCmd is a command that evaluates an expression for the given time (range)
/// and expects a specific result.
pub(crate) struct EvalCmd {
    expr : String,
    start: timestamp,
    line: usize,
    fail: bool,
    ordered: bool,
    metrics: BTreeMap<u64, MetricName>,
    expected: BTreeMap<u64, Entry>
}

#[derive(Debug, Clone)]
struct Entry {
    pos: usize,
    vals: Vec<SequenceValue>
}

impl Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.pos, self.vals.join(","))
    }
}

impl EvalCmd {
    fn new(expr: String, start: timestamp, line: usize) -> EvalCmd {
        return EvalCmd{
            expr,
            start,
            line,
            fail: false,
            ordered: false,
            metrics:  Default::default(),
            expected: Default::default(),
        }
    }

    /// expect adds a new metric with a sequence of values to the set of expected
    /// results for the query.
    pub fn expect(&mut self, pos: usize, m: MetricName, vals: &[SequenceValue]) {
        if m.is_empty() {
            self.expected.set(0, Entry{ pos, vals });
            return
        }
        let h = m.hash();
        self.metrics.set(h, m);
        self.expected.set(h, Entry{ pos, vals });
    }

    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        let mut queries = at_modifier_test_cases(self.expr, self.start)?;

        queries.push(AtModifierTestCase{
            expr: self.expr,
            eval_time: self.start
        });

        for iq in queries.iter() {
            let q = self.query_engine.new_instant_query(self.storage, nil, iq.expr, iq.eval_time)?;

            let res = q.exec(self.context);
            if res.err.is_some() {
                if cmd.fail {
                    continue
                }
                let msg = format!("error evaluating query {} (line {}): {:?}", iq.expr, cmd.line, res.err);
                return Err(RuntimeError::from(msg))
            }

            if res.err.is_some() && cmd.fail {
                let msg = format!("expected error evaluating query {} (line {}) but got none", iq.expr, cmd.line);
                return Err(RuntimeError::from(msg))
            }

            if let Err(err) = self.compare_result(res.value) {
                let msg = format!("error in {} {}: {:?}", cmd, iq.expr, err);
                return Err(RuntimeError::from(msg))
            }

            // Check query returns same result in range mode,
            // by checking against the middle step.
            let q = self.queryEngine.new_range_query(self.storage, iq.expr,
                iq.eval_time.add(-time.Minute),
                iq.eval_time.add(time.Minute),
                time.Minute)?;

            let range_res = q.exec(t.context);

            if range_res.err.is_some() {
                let msg = format!("error evaluating query {} (line {}) in range mode: {}", 
                    iq.expr, self.line, range_res.err);
                return Err(RuntimeError::from(msg));
            }

            if cmd.ordered {
                // Ordering isn't defined for range queries.
                continue
            }
            let mat = range_res.value.as_range_vector();

            let vec = Vec::with_capacity(mat.len());
            for series in mat {
                for point in series.points.iter() {
                    if point.t == iq.eval_time {
                        vec.push(Sample{metric: series.metric, point});
                        break
                    }
                }
            }

            if _, ok := res.Value.(Scalar); ok {
                err = self.compare_result(Scalar{V: vec[0].point.v})
            } else {
                err = self.compare_result(vec)
            }
            if err != nil {
                return format!("error in {} {} (line {}) range mode: {}", cmd, iq.expr, cmd.line, err)
            }

        }
    }

    /// compare_result compares the result value with the defined expectation.
    fn compare_result(&self, result: AnyValue) -> RuntimeResult<()> {
        match result {
            QueryValue::RangeVector(_) => {
                return Err(RuntimeError::from("received range result on instant evaluation"));
            },
            QueryValue::InstantVector(vector) => {
                let seen: BTreeSet<u64> = Default::default();
                for (pos, v) in val.iter().enumerate() {
                    let fp = self.metric.hash();
                    if !ev.metrics.contains_key(fp) {
                        return fmt.Errorf("unexpected metric {} in result", v.metric);
                    }
                    let exp = self.expected.get(fp).unwrap(); // todo: expect()
                    if self.ordered && exp.pos != pos+1 {
                        let msg = format!("expected metric {} with {:?} at position {} but was at {}",
                                          self.metric, self.vals, exp.pos, pos+1);
                        return Err(RuntimeError::from(msg));
                    }
                    if !almost_equal(exp.vals[0].value, v.v) {
                        let msg = format!("expected {} for {} but got {}", exp.vals[0].value, v.metric, v.v);
                        return Err(RuntimeError::from(msg));
                    }
                    seen.insert(fp);
                }
                for (fp, exp_vals) in self.expected.iter() {
                    if !seen.contains(fp) {
                        fmt.Println("vector result", val.len(), ev.expr);
                        for ss in val.iter() {
                            fmt.println("    ", ss.metric, ss.point);
                        }
                        let metric = self.metrics.get(fp);
                        let msg = format!("expected metric {} with {:?} not found",
                                          metric, exp_vals);

                        return Err(RuntimeError::from(msg));
                    }
                }
            },
            QueryValue::Scalar(v) => {
                if !almost_equal(self.expected[0].vals[0].value, val.v) {
                    let msg = format!("expected Scalar {} but got {}",
                                      val.v, ev.expected[0].vals[0].value);
                    return Err(RuntimeError::from(msg));
                }
            },
            _ => {
                panic!("promql.Test.compare_result: unexpected result type {:?}", result)
            }
        }

        Ok(())
    }
}


impl Display for EvalCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval {} @ {}", self.expr, self.start)
    }
}


/// clearCmd is a command that wipes the test's storage state.
pub(crate) struct ClearCmd {}

impl ClearCmd {
    pub fn new() -> Self {
        ClearCmd {}
    }

    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        t.clear();
    }
}

impl Display for ClearCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "clear()")
    }
}


struct AtModifierTestCase {
    expr: String,
    eval_time: Timestamp
}

fn at_modifier_test_cases(expr_str: String, eval_time: timestamp) -> RuntimeResult<Vec<AtModifierTestCase>> {
    let expr = parse(exprStr)?;
    let ts = timestamp.FromTime(eval_time);

    let contains_non_step_invariant = false;
    // Setting the @ timestamp for all selectors to be eval_time.
    // If there is a subquery, then the selectors inside it don't get the @ timestamp.
    // If any selector already has the @ timestamp set, then it is untouched.
    parser.inspect(expr, |node: Node, path: &[Node]| -> RuntimeResult<()> {
        let subqTs = subqueryTimes(path);
        if subqTs != nil {
            // There is a subquery with timestamp in the path,
            // hence don't change any timestamps further.
            return nil
        }
        match node {
            Expression::MetricExpression(me) => {
                if n.timestamp == nil {
                    n.timestamp = makeInt64Pointer(ts)
                }

            Expression::MatrixSelector(me) => {
                if vs = n.VectorSelector.(*parser.VectorSelector);
                    vs.timestamp == 0 {
                        vs.timestamp = makeInt64Pointer(ts)
                    }
                }


            Expression::RollupExpr(_) => {
                if n.timestamp == nil {
                    n.timestamp = makeInt64Pointer(ts)
                }
            }
        },
    Expression::Function(fe) => {
        ok := AtModifierUnsafeFunctions[n.Func.Name];
        contains_non_step_invariant = contains_non_step_invariant || ok
    }
})

    if contains_non_step_invariant {
        // Since there is a step invariant function, we cannot automatically
        // generate step invariant test cases for it sanely.
        return nil
    }

    let new_expr = expr.to_string(); // With all the @ eval_time set.
    let c = &[-10 * ts, 0, ts / 5, ts, 10 * ts];
    if ts == 0 {
        additional_eval_times = [-1000, -ts, 1000];
    }
    let test_cases = Vec::with_capacity(additional_eval_times.len());
    for et in additional_eval_times.iter() {
        test_cases.push(AtModifierTestCase{
            expr: new_expr,
            eval_time: timestamp.Time(et),
        })
    }
    return test_cases
}

fn parse_number(s: &str) -> RuntimeResult<f64> {
    match i64::try_from(s) {
        Ok(v) => v as f64,
        Err(e) => {
            return match f64::try_from(s) {
                Ok(v) => Ok(v),
                Err(e) => {
                    Err(RuntimeError::from(format!("error parsing number: {}", s)))
                }
            }
        }
    }
}

/// LazyLoader lazily loads samples into storage.
/// This is specifically implemented for unit testing of rules.
pub(crate) struct LazyLoader {
    load_cmd: LoadCmd,
    storage: Storage,
    subquery_interval: Duration,
    query_engine: Engine,
    context: Context,
    cancel_ctx: CancelFunc,
    opts: LazyLoaderOpts
}

/// LazyLoaderOpts are options for the lazy loader.
pub struct LazyLoaderOpts {
    /// Both of these must be set to true for regular PromQL (as of
    /// Prometheus v2.33). They can still be disabled here for legacy and
    /// other uses.
    enable_at_modifier: bool,
    enable_negative_offset: bool
}

impl LazyLoader {
    /// new returns an initialized empty LazyLoader.
    fn new(input: &str, opts: LazyLoaderOpts) -> RuntimeResult<LazyLoader> {
        let ll = LazyLoader {
            load_cmd: LoadCmd {
                gap: Default::default(),
                metrics: Default::default(),
                defs: Default::default(),
            },
            storage: (),
            subquery_interval: Default::default(),
            query_engine: (),
            context: (),
            cancel_ctx: (),
            opts,
        };

        ll.parse(input)?;
        ll.clear();

        Ok(ll)
    }

    /// parse the given load command.
    fn parse(&mut self, input: &str) -> RuntimeResult<()> {
        let lines = get_lines(input);
        /// Accepts only 'load' command.
        for line in lines.iter() {
            if line.is_empty() {
                continue;
            }
            if strings.ToLower(patSpace.split(l, 2)[0]) == "load" {
                self.load_cmd = parse_load(lines, i)?;
                Ok(())
            }
            return raise(i, format!("invalid command {}", l));
        }
        return RuntimeError::from("no \"load\" command found")
    }

    // clear the current test storage of all inserted samples.
    fn clear(&mut self) {
        match self.storage.close() {
            Ok(_) => {},
            Err(e) => {
                return Err(RuntimeError::from("Unexpected error while closing test storage."))
            }
        }
        if let Some(cancel_func) = self.cancel_ctx {
            (cancel_func)();
        }
        self.storage = TestStorage::new(ll);

        let opts = EngineOpts {
            max_samples: 10000,
            timeout: 100 * time.Second,
            NoStepSubqueryIntervalFn: Duration::from_millis(self.SubqueryInterval)
        };

        self.queryEngine = NewEngine(opts)
            (self.context, self.cancelCtx) = context.WithCancel(context.Background())
    }

    /// append_till appends the defined time series to the storage till the given timestamp
    /// (in milliseconds).
    fn append_till(&mut self, ts: i64) -> RuntimeResult<()> {
        let app = self.storage.get_appender(self.context);
        let to_remove: Vec<u64> = vec![];

        for (h, samples) in self.load_cmd.defs {
            let m = self.load_cmd.metrics.get(h);
            for (i, s) in samples.iter() {
                if s.t > ts {
                    // Removing the already added samples.
                    self.load_cmd.defs[h] = &samples[i..];
                    break
                }
                app.append(0, m, s.t, s.v)?;
                if i == samples.len() - 1 {
                    self.load_cmd.defs[h] = nil
                }
            }
        }
        app.commit()
    }

    // Close closes resources associated with the LazyLoader.
    pub fn close(&mut self) -> RuntimeResult<()> {
        self.cancelCtx();
        self.storage.close();
        require.NoError(ll.T, err, "Unexpected error while closing test storage.")
    }

    // with_samples_till loads the samples till given timestamp and executes the given function.
    fn with_samples_till(&mut self, ts: timestamp, func: fn(RuntimeError)) {
        let ts_milli = ts.Sub(time.Unix(0, 0).UTC()) / time.Millisecond;
        func(self.append_till(ts_milli as i64));
    }

    /// Queryable allows querying the LazyLoader's data.
    /// Note: only the samples till the max timestamp used
    /// in `with_samples_till` can be queried.
    pub fn queryable(&self) -> &Queryable {
        &self.storage
    }
}


struct SeriesDescription {
    labels: MetricName,
    values: Vec<SequenceValue>
}

// ParseSeriesDesc parses the description of a time series.
fn parse_series_desc(input: &str) -> RuntimeResult<SeriesDescription> {
    let p = newParser(input);

    let parse_result = p.parse_generated(START_SERIES_DESCRIPTION)?;
    let result = parse_result.(*seriesDescription)
    labels = result.labels
    values = result.values

    return SeriesDescription{ labels, values }
}