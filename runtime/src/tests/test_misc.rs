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

use std::collections::btree_map::BTreeMap;
use std::fmt::{Display, Formatter};
use std::time::Duration;

const
MIN_NORMAL: f64 = f64::from_bits(0x0010000000000000); // The smallest positive normal value of type float64.

const PATTERN_SPACE: &str = "[\t ]+";
const PATTERN_LOAD: &str = r"^load\s+(.+?)$";
const PATTERN_EVAL_INSTANT: &str = r"^eval(?:_(fail|ordered))?\s+instant\s+(?:at\s+(.+?))?\s+(.+)$";


/// Relative error allowed for sample values.
const EPSILON: f64 = 0.000001;


/// Point represents a single data point for a given timestamp.
/// If H is not nil, then this is a histogram point and only (T, H) is valid.
/// If H is nil, then only (T, V) is valid.
#[derive(Debug, Default)]
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

// Test is a sequence of read and write commands that are run
// against a test storage.
pub(crate) struct Test {
    cmds: Vec<TestCommand>,
    storage: TestStorage,
    query_engine: Engine,
    context: Context,
    cancel_ctx: CancelFunc
}

impl Test {

    // NewTest returns an initialized empty Test.
    pub fn new(input: String) -> RuntimeResult<Test> {
        let test = Test{
            cmds: vec![],
            storage: (),
            query_engine: (),
            context: (),
            cancel_ctx: ()
        };
        test.parse(input)?;
        test.clear();

        Ok(test)
    }

    pub fn from_file(filename: String) -> RuntimeResult<Test> {
        let content = os.ReadFile(filename)?;
        return Self::new(content)
    }

    /// parse the given command sequence and appends it to the test.
    fn parse(&mut self, input: String) -> RuntimeResult<()> {
        let lines = getLines(input);
        // Scan for steps line by line.
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let cmd_str = Strings::toLower(patSpace.Split(l, 2)[0]);
            let cmd: TestCommand;

            match cmd_str {
                "clear" => cmd = ClearCmd::new(),
                "load" => {
                    (i, cmd) = parse_load(lines, i)?
                },
                _ => {
                    if cmd_str.starts_with("eval") {
                        (i, cmd) = parse_eval(lines, i)?;
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
            err = t.storage.Close();
            require.NoError(t.T, err, "Unexpected error while closing test storage.")
        }
        if let Some(cancel_ctx) = self.cancel_ctx {
            (cancel_ctx)()
        }
        self.storage = teststorage.New(t);
        let opts = EngineOpts {
            MaxSamples: 10000,
            Timeout: 100 * time.Second,
            NoStepSubqueryIntervalFn: Duration::from_millis(1000 * 60)
        };

        self.queryEngine = NewEngine(opts)
            (self.context, self.cancelCtx) = context.WithCancel(context.Background())
    }

    // Close closes resources associated with the Test.
    fn close(&mut self) -> RuntimeResult<()> {
        self.cancel_ctx();
        err := t.storage.close();
        require.NoError(t.T, err, "Unexpected error while closing test storage.")
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
}



fn raise(line: usize, format: String, v ...interface{}) -> RuntimeResult<()> {
    return &parser.ParseErr{
        lineOffset: line,
        err: fmt.Errorf(format, v...),
    }
}

fn parse_load(lines: &[String], i: usize) -> RuntimeResult<(usize, LoadCmd)> {
    if !patLoad.MatchString(lines[i]) {
        return raise(i, "invalid load command. (load <step:duration>)")
    }
    let parts = patLoad.FindStringSubmatch(lines[i]);

    let gap = duration_value(parts[1], 1);
    if let Err(err) = gap {
        return raise(i, "invalid step definition {}: {}", parts[1], err)
    }
    let gap = gap.unwrap();
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
                perr.LineOffset = i
            }
            return Err(err)
        }
        cmd.set(metric, vals...)
    }
    return Ok((i, cmd))
}

fn parse_eval(t: &mut Test, lines: &[String], i: usize) -> RuntimeResult<(usize, EvalCmd)> {
    if !patEvalInstant.MatchString(lines[i]) {
        return raise(i, "invalid evaluation command. (eval[_fail|_ordered] instant [at <offset:duration>] <query>")
    }
    let parts = patEvalInstant.FindStringSubmatch(lines[i]);
    let mod_  = parts[1];
    let at   = parts[2];
    let expr = parts[3];

    _, err = parser.ParseExpr(expr);
    if err != nil {
        let perr: ParseErr;
        if errors.As(err, &perr) {
            perr.LineOffset = i;
            let pos_offset = parser.Pos(strings.Index(lines[i], expr));
            perr.span.start += pos_offset;
            perr.span.end += pos_offset;
            perr.query = lines[i]
        }
        return Err(err)
    }

    offset, err := model.parse_duration(at);
    if err != nil {
        return raise(i, "invalid step definition {}: %s", parts[1], err)
    }
    let ts = testStartTime.Add(time.Duration(offset));
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
            i--
            break
        }
        if f, err := parseNumber(def_line); err == nil {
            self.expect(0, nil, SequenceValue{Value: f})
            break
        }
        metric, vals, err = parse_series_desc(defLine);
        if err != nil {
            let perr: ParseErr;
            if errors.As(err, &perr) {
                perr.LineOffset = i
            }
            return Err(err)
        }

        // Currently, we are not expecting any matrices.
        if vals.len() > 1 {
            raise(i, "expecting multiple values in instant evaluation not allowed");
        }
        cmd.expect(j, metric, vals...)
    }
    return Ok((i, cmd))
}

// get_lines returns trimmed lines after removing the comments.
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
struct SequenceValue {
    pub value: f64,
    pub omitted: bool
}

impl Display for SequenceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.omitted {
            write(f, "_")
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
        let app = t.storage.get_appender(t.context);
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
        let h = m.Hash();
        let samples: Vec<Point> = Vec::with_capacity(vals.len());
        let ts = testStartTime;
        for v in vals.iter() {
            if !v.omitted {
                samples.push(Point{
                        t: ts.UnixNano() / int64(time.Millisecond/time.Nanosecond),
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
    start: Timestamp,
    line: usize,
    fail: bool,
    ordered: bool,
    metrics: BTreeMap<u64, MetricName>,
    expected: BTreeMap<u64, Entry>
}

#[derive(Debug)]
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
    fn new(expr: String, start: Timestamp, line: usize) -> EvalCmd {
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
    fn expect(&mut self, pos: usize, m: Labels, vals: &[SequenceValue]) {
        if m == nil {
            self.expected[0] = Entry{ pos, vals };
            return
        }
        let h = m.Hash();
        ev.metrics[h] = m;
        ev.expected[h] = Entry{ pos, vals  }
    }

    fn exec(&mut self, t: &mut Test) -> RuntimeResult<()> {
        let mut queries = at_modifier_test_cases(self.expr, self.start)?;

        queries = append([]atModifierTestCase{{expr: self.expr, eval_time: self.start}}, queries...)
        for iq in queries.iter() {
            let q = t.query_engine.new_instant_query(t.storage, nil, iq.expr, iq.eval_time)?;

            let res = q.exec(t.context);
            if res.Err != nil {
                if cmd.fail {
                    continue
                }
                const msg = format!("error evaluating query {} (line {}): {:?}", iq.expr, cmd.line, res.Err);
                return Err(RuntimeError::from(msg))
            }

            if res.Err == nil && cmd.fail {
                let msg = format!("expected error evaluating query {} (line {}) but got none", iq.expr, cmd.line);
                return Err(RuntimeError::from(msg))
            }

            if let Err(err) = self.compare_result(res.value) {
                let msg = format!("error in {} {}: {:?}", cmd, iq.expr, err);
                return Err(RuntimeError::from(msg))
            }

            // Check query returns same result in range mode,
            // by checking against the middle step.
            let q, err = t.queryEngine.new_range_query(t.storage, nil, iq.expr,
                iq.eval_time.Add(-time.Minute),
                iq.eval_time.Add(time.Minute),
                time.Minute)

            let rangeRes = q.exec(t.context);

            if rangeRes.Err != nil {
                let msg = format!("error evaluating query {} (line {}) in range mode: {}", iq.expr, self.line, rangeRes.Err)
                return Err(RuntimeError::from(msg));
            }

            if cmd.ordered {
                // Ordering isn't defined for range queries.
                continue
            }
            let mat = rangeRes.Value.(Matrix)

            let vec = Vec::with_capacity(mat.len())
            for series in mat {
                for point in series.points.iter() {
                    if point.t == timeMilliseconds(iq.eval_time) {
                        vec.push(Sample{metric: series.metric, point})
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
                return fmt.Errorf("error in {} {} (line {}) range mode: {}", cmd, iq.expr, cmd.line, err)
            }

        }
    }

    /// compare_result compares the result value with the defined expectation.
    fn compare_result(&self, result: AnyValue) -> RuntimeResult<()> {
        match result {
            AnyValue::RangeVector(_) => {
                return Err(RuntimeError::from("received range result on instant evaluation"));
            },
            AnyValue::InstantVector(vector) => {
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
                            fmt.Println("    ", ss.metric, ss.point);
                        }
                        let metric = self.metrics.get(fp);
                        let msg = format!("expected metric {} with {:?} not found",
                                          metric, exp_vals);

                        return Err(RuntimeError::from(msg));
                    }
                }
            },
            AnyValue::Scalar(v) => {
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

fn at_modifier_test_cases(expr_str: String, eval_time: Timestamp) -> RuntimeResult<Vec<AtModifierTestCase>> {
    let expr = parser.ParseExpr(exprStr)?;
    let ts = timestamp.FromTime(eval_time);

    let contains_non_step_invariant = false;
    // Setting the @ timestamp for all selectors to be eval_time.
    // If there is a subquery, then the selectors inside it don't get the @ timestamp.
    // If any selector already has the @ timestamp set, then it is untouched.
    parser.inspect(expr, |node: Node, path: &[Node]| -> RuntimeResult<()> {
        _, _, subqTs = subqueryTimes(path);
if subqTs != nil {
// There is a subquery with timestamp in the path,
// hence don't change any timestamps further.
return nil
}
        match node {
            Expression::MetricExpression(me) => {
            if n.timestamp == nil {
                n.Timestamp = makeInt64Pointer(ts)
            }

            Expression::MatrixSelector(me) => {
                    if vs: = n.VectorSelector.(*parser.VectorSelector);
                    vs.Timestamp == 0 {
                        vs.timestamp = makeInt64Pointer(ts)
                    }
                }


            Expression::RollupExpr:
            if n.Timestamp == nil {
                n.Timestamp = makeInt64Pointer(ts)
            }
        },
    Expression::Function(fe) => {
        _, ok := AtModifierUnsafeFunctions[n.Func.Name]
        contains_non_step_invariant = contains_non_step_invariant || ok
    }
return nil
})

    if contains_non_step_invariant {
        // Since there is a step invariant function, we cannot automatically
        // generate step invariant test cases for it sanely.
        return nil
    }

    let new_expr = expr.to_string(); // With all the @ eval_time set.
    additionaleval_times = &[-10 * ts, 0, ts / 5, ts, 10 * ts];
    if ts == 0 {
        additionaleval_times = [-1000, -ts, 1000];
    }
    let test_cases = Vec::with_capacity(additionaleval_times.len());
    for et in additionaleval_times.iter() {
        test_cases.push(AtModifierTestCase{
            expr: new_expr,
            eval_time: timestamp.Time(et),
        })
    }
    return test_cases
}


/// returns true if the two sample lines only differ by a
/// small relative error in their sample value.
fn almost_equal(a: f64, b: f64) -> bool {
    // NaN has no equality but for testing we still want to know whether both values
    // are NaN.
    if a.is_nan() && b.is_nan() {
        return true
    }

    // Cf. http://floating-point-gui.de/errors/comparison/
    if a == b {
        return true
    }

    let diff = (a - b).abs();

    if a == 0 || b == 0 || diff < MIN_NORMAL {
        return diff < EPSILON * MIN_NORMAL
    }
    
    return diff/(a.abs() + b.abs()) < EPSILON
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

// LazyLoaderOpts are options for the lazy loader.
pub struct LazyLoaderOpts {
    // Both of these must be set to true for regular PromQL (as of
    // Prometheus v2.33). They can still be disabled here for legacy and
    // other uses.
    enable_at_modifier: bool,
    enable_negative_offset: bool
}

impl LazyLoader {
    /// new returns an initialized empty LazyLoader.
    fn new(input: &str, opts: LazyLoaderOpts) -> RuntimeResult<LazyLoader> {
        let ll = LazyLoader {
            load_cmd: LoadCmd {},
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
            if strings.ToLower(patSpace.Split(l, 2)[0]) == "load" {
                self.load_cmd = parse_load(lines, i)?;
                Ok(())
            }
            return raise(i, "invalid command {}", l);
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
        self.storage = teststorage.New(ll);

        let opts = EngineOpts {
            MaxSamples: 10000,
            Timeout: 100 * time.Second,
            NoStepSubqueryIntervalFn: Duration::from_millis(self.SubqueryInterval),
            EnableAtModifier: self.opts.EnableAtModifier,
            EnableNegativeOffset: self.opts.EnableNegativeOffset,
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
                    self.load_cmd.defs[h] = samples
                    [i: ]
                    break
                }
                app.append(0, m, s.t, s.v)?;
                if i == samples.len() - 1 {
                    self.load_cmd.defs[h] = nil
                }
            }
        }
        return app.commit()
    }

    // Close closes resources associated with the LazyLoader.
    pub fn close(&mut self) -> RuntimeResult<()> {
        self.cancelCtx();
        err: = self.storage.close();
        require.NoError(ll.T, err, "Unexpected error while closing test storage.")
    }

    // with_samples_till loads the samples till given timestamp and executes the given function.
    fn with_samples_till(&mut self, ts: Timestamp, func: fn(RuntimeError)) {
        let ts_milli = ts.Sub(time.Unix(0, 0).UTC()) / time.Millisecond;
        func(self.append_till(int64(ts_milli)));
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

    let parse_result = p.parseGenerated(START_SERIES_DESCRIPTION);
    if parse_result != nil {
        let result = parse_result.(*seriesDescription)
        labels = result.labels
        values = result.values
    }

    if len(p.parseErrors) != 0 {
        err = p.parseErrors
    }

    return SeriesDescription{ labels, values }
}