use std::collections::{BTreeMap, BTreeSet, Vec};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use regex::Regex;

use metricsql::{pushdown_binary_op_filters, trim_filters_by_group_modifier};
use metricsql::types::*;

use crate::{EvalConfig, MetricName};
use crate::tag_filter::{TagFilter, to_tag_filters};
use crate::timeseries::Timeseries;
use crate::runtime_error::{RuntimeError, RuntimeResult};

use crate::binary_op::*;
use crate::eval::{create_evaluator};
use crate::eval::traits::Evaluator;

#[derive(Debug, Clone, PartialEq, Default)]
pub(self) struct BinaryOpEvaluator {
    expr: BinaryOpExpr,
    binop_fn: &'static BinaryOpFunc,
    lhs: Box<dyn Evaluator>,
    rhs: Box<dyn Evaluator>,
    can_pushdown_filters: bool
}

impl BinaryOpEvaluator {
    pub fn new(expr: &BinaryOpExpr) -> Self {
        let binop_fn = get_binary_op_func(be.op)?;
        let lhs = Box::new( create_evaluator(&expr.left) );
        let rhs = Box::new( create_evaluator(&expr.right) );
        let can_pushdown_filters = can_pushdown_common_filters(expr);
        Self {
            binop_fn,
            lhs,
            rhs,
            expr: expr.clone(),
            can_pushdown_filters
        }
    }

    fn eval_args(&self, ec: &mut EvalConfig) -> RuntimeResult<(Vec<Timeseries>, Vec<Timeseries>)> {

        if !self.can_pushdown_filters {
            // Execute both sides in parallel, since it is impossible to push down common filters
            // from exprFirst to exprSecond.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886

            // todo: we need to clone ec
            let (tss_first, tss_second) = match rayon::join(
                || self.lhs.eval(ec),
                || self.rhs.eval(ec),
            ) {
                (Err(err), _) | (Ok(_), Err(err)) => return Err(err),
                (Ok(Some(first)), Ok(Some(second))) => (first, second),
            };

            return Ok(tss_first, tss_second)
        }

        /// Execute binary operation in the following way:
        ///
        /// 1) execute the expr_first
        /// 2) get common label filters for series returned at step 1
        /// 3) push down the found common label filters to expr_second. This filters out unneeded series
        ///    during expr_second execution instead of spending compute resources on extracting and processing these series
        ///    before they are dropped later when matching time series according to https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
        /// 4) execute the expr_second with possible additional filters found at step 3
        ///
        /// Typical use cases:
        /// - Kubernetes-related: show pod creation time with the node name:
        ///
        ///     kube_pod_created{namespace="prod"} * on (uid) group_left(node) kube_pod_info
        ///
        ///   Without the optimization `kube_pod_info` would select and spend compute resources
        ///   for more time series than needed. The selected time series would be dropped later
        ///   when matching time series on the right and left sides of binary operand.
        ///
        /// - Generic alerting queries, which rely on `info` metrics.
        ///   See https://grafana.com/blog/2021/08/04/how-to-use-promql-joins-for-more-effective-queries-of-prometheus-metrics-at-scale/
        ///
        /// - Queries, which get additional labels from `info` metrics.
        ///   See https://www.robustperception.io/exposing-the-software-version-to-prometheus

        let tss_first = self.lhs.eval(ec)?;
        let mut second = &self.expr.left;

        if self.expr.op != BinaryOp::Or {
            // do not pushdown common label filters from tss_first for `or` operation, since this can filter out
            // the needed time series from tss_second.
            // See https://prometheus.io/docs/prometheus/latest/querying/operators/#logical-set-binary-operators for details.
            let mut lfs = get_common_label_filters(&tss_first[0..]);
            trim_filters_by_group_modifier(&mut lfs, &self.expr);
            second = &pushdown_binary_op_filters(&self.expr.right, lfs)
        }
        let tss_second = second(ec)?;
        return Ok((tss_first, tss_second));
    }

}

impl Evaluate for BinaryOpEvaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ec: &mut EvalConfig) -> RuntimeResult<&Vec<Timeseries>> {
        let tss_left: Vec<Timeseries> = vec![];
        let tss_right: Vec<Timeseries> = vec![];

        if &self.expr.op == BinaryOp::And || &self.expr.op == BinaryOp::If {
            // Fetch right-side series at first, since it usually contains
            // lower number of time series for `and` and `if` operator.
            // This should produce more specific label filters for the left side of the query.
            // This, in turn, should reduce the time to select series for the left side of the query.
            (tss_right, tss_left) = self.eval_args(ec)?
        } else {
            (tss_left, tss_right) = self.eval_args(ec)?
        }

        let bfa = BinaryOpFuncArg {
            be: self.expr,
            left,
            right,
        };

        match self.binop_fn(bfa) {
            Err(err) => Err(RuntimeError::from(format!("cannot evaluate {}: {}", &self.expr, err))),
            OK(v) => Ok(v)
        }
    }
}

fn can_pushdown_common_filters(be: &BinaryOpExpr) -> bool {
    match be.op {
        BinaryOp::Or | BinaryOp::Default => false,
        _=> {
            if is_aggr_func_without_grouping(&be.left) || 
                is_aggr_func_without_grouping(&be.right) {
                return false
            }
            return true
        }
    }
}

fn is_aggr_func_without_grouping(e: &BExpression) -> bool {
    match e {
        Expression::Aggregate(afe) => {
            afe.modifier.args.len == 0
        },
        _ => false
    }
}

fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    let mut m: BTreeMap<String, BTreeSet<String>> = BTreeMap::with_capacity(tss.len());
    for ts in tss.iter() {
        for (key, value) in ts.metric_name.iter().enumerate() {
            if let Some(set) = m.get_mut(key) {
                set.insert(value)
            } else {
                let s = BTreeSet::from([value]);
                m.insert(key, s);
            }
        }
    }

    let mut lfs: Vec<LabelFilter> = Vec::with_capacity(m.len());
    for (key, values) in m {
        if values.len() != tss.len() {
            // Skip the tag, since it doesn't belong to all the time series.
            continue;
        }
        if values.len() > 1000 {
            // Skip the filter on the given tag, since it needs to enumerate too many unique values.
            // This may slow down the search for matching time series.
            continue;
        }

        let vals = values.iter().collect().sort();
        let mut str_value: String;
        let mut is_regex = false;

        if values.len() == 1 {
            str_value = values[0]
        } else {
            str_value = join_regexp_values(vals);
            is_regex = true;
        }
        let lf = if is_regex {
            LabelFilter::equal(key, str_value)?
        } else {
            LabelFilter::regex_equal(key, str_value)?
        };

        lfs.push(lf);
    }
    lfs.sort();
    return lfs;
}


fn join_regexp_values(a: &Vec<String>) -> String {
    let init_size = a.iter().fold(
        0,
        |res, x| res + x.len(),
    );
    let mut res = String::with_capacity(init_size);
    for (i, s) in a.iter().enumerate() {
        let s_quoted = Regex::quote(s);
        res.push_str(s_quoted);
        if i < a.len() - 1 {
            b.push('|')
        }
    }
    return res;
}
