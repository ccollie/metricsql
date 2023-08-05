use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Display;

use crate::{MetricName, RuntimeError, RuntimeResult, Sample, Timestamp};
use crate::functions::types::AnyValue;
use crate::tests::helpers::Labels;
use crate::tests::test::{at_modifier_test_cases, AtModifierTestCase, Test};
use crate::tests::types::SequenceValue;
use crate::tests::utils::almost_equal;

/// EvalCmd is a command that evaluates an expression for the given time (range)
/// and expects a specific result.
pub(crate) struct EvalCmd {
    expr: String,
    start: Timestamp,
    line: usize,
    metrics: BTreeMap<u64, MetricName>,
    expected: BTreeMap<u64, Entry>,
    pub fail: bool,
    pub ordered: bool,
}

#[derive(Debug, Clone)]
struct Entry {
    pos: usize,
    vals: Vec<SequenceValue>,
}


impl Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.pos, self.vals.join(","))
    }
}

impl EvalCmd {
    pub(crate) fn new(expr: &str, start: Timestamp, line: usize) -> EvalCmd {
        return EvalCmd {
            expr: expr.to_string(),
            start,
            line,
            fail: false,
            ordered: false,
            metrics: Default::default(),
            expected: Default::default(),
        }
    }

    /// expect adds a new metric with a sequence of values to the set of expected
    /// results for the query.
    pub fn expect(&mut self, pos: usize, m: Labels, vals: &[SequenceValue]) {
        if m.is_empty() {
            self.expected.set(0, Entry { pos, vals });
            return
        }
        let h = m.hash();
        self.metrics.set(h, m);
        self.expected.set(h, Entry { pos, vals });
    }

    pub(crate) fn exec(&mut self, test: &mut Test) -> RuntimeResult<()> {
        let mut queries = at_modifier_test_cases(&self.expr, self.start)?;

        queries.push(AtModifierTestCase {
            expr: self.expr.clone(),
            eval_time: self.start,
        });

        for iq in queries.iter() {
            let q = self.query_engine.new_instant_query(self.storage,
                                                        nil,
                                                        iq.expr,
                                                        iq.eval_time)?;

            let res = q.exec(self.context);
            if res.err.is_some() {
                if self.fail {
                    continue
                }
                let msg = format!("error evaluating query {} (line {}): {:?}", iq.expr, cmd.line, res.err);
                return Err(RuntimeError::from(msg))
            }

            if res.err.is_some() && self.fail {
                let msg = format!("expected error evaluating query {} (line {}) but got none", iq.expr, cmd.line);
                return Err(RuntimeError::from(msg))
            }

            if let Err(err) = self.compare_result(res.value) {
                let msg = format!("error in {cmd} {}: {:?}", iq.expr, err);
                return Err(RuntimeError::from(msg))
            }

            // Check query returns same result in range mode,
            // by checking against the middle step.
            let q = self.queryEngine.new_range_query(self.storage, iq.expr,
                                                          iq.eval_time.add(-time.Minute),
                                                          iq.eval_time.add(time.Minute),
                                                          time.Minute)

            let range_res = q.exec(t.context);

            if range_res.err.is_some() {
                let msg = format!("error evaluating query {} (line {}) in range mode: {}",
                                  iq.expr, self.line, range_res.err);
                return Err(RuntimeError::from(msg));
            }

            if self.ordered {
                // Ordering isn't defined for range queries.
                continue
            }
            let mat = range_res.Value.(Matrix)

            let mut vec = Vec::with_capacity(mat.len());
            for series in mat {
                for point in series.points.iter() {
                    if point.t == timeMilliseconds(iq.eval_time) {
                        vec.push(Sample { metric: series.metric, point });
                        break
                    }
                }
            }

            if let QueryValue::Scalar(v) = res.Value {
                if vec.len() != 1 {
                    let msg = format!("expected 1 result for query {} (line {}) but got {}",
                                      iq.expr, self.line, vec.len());
                    return Err(RuntimeError::from(msg));
                }
            } else {
                self.compare_result(vec)
            }
            if err != nil {
                return format!("error in {} {} (line {}) range mode: {}", cmd, iq.expr, cmd.line, err)
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
                let mut seen: BTreeSet<u64> = Default::default();
                for (pos, v) in vector.iter().enumerate() {
                    let fp = self.metric.hash();
                    if !v.metric_name.contains_key(fp) {
                        return fmt.Errorf("unexpected metric {} in result", v.metric);
                    }
                    let exp = self.expected.get(fp).unwrap(); // todo: expect()
                    if self.ordered && exp.pos != pos + 1 {
                        let msg = format!("expected metric {} with {:?} at position {} but was at {}",
                                          self.metric, self.vals, exp.pos, pos + 1);
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
                        println!("vector result", val.len(), ev.expr);
                        for ss in val.iter() {
                            fmt.println("    ", ss.metric, ss.point);
                        }
                        let metric = self.metrics.get(fp);
                        let msg = format!("expected metric {metric} with {:?} not found",
                                          exp_vals);

                        return Err(RuntimeError::from(msg));
                    }
                }
            },
            AnyValue::Scalar(v) => {
                if !almost_equal(self.expected[0].vals[0].value, v) {
                    let msg = format!("expected Scalar {} but got {}",
                                      v, ev.expected[0].vals[0].value);
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
