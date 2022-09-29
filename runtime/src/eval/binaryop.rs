use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;

use regex::escape;

use metricsql::ast::*;
use metricsql::functions::Volatility;
use metricsql::optimizer::{
    can_pushdown_op_filters,
    pushdown_binary_op_filters,
    trim_filters_by_group_modifier
};

use crate::binary_op::{BinaryOpFn, BinaryOpFuncArg, get_binary_op_handler};
use crate::context::Context;
use crate::eval::{create_evaluator, ExprEvaluator};
use crate::eval::traits::Evaluator;
use crate::EvalConfig;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::timeseries::Timeseries;

pub(super) struct BinaryOpEvaluator {
    expr: BinaryOpExpr,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    handler: Box<dyn BinaryOpFn>,
    can_pushdown_filters: bool,
    can_pushdown_op_filters: bool
}

impl BinaryOpEvaluator {
    pub fn new(expr: &BinaryOpExpr) -> RuntimeResult<Self> {
        let lhs = Box::new( create_evaluator(&expr.left)? );
        let rhs = Box::new(create_evaluator(&expr.right)? );
        let can_pushdown_filters = can_pushdown_common_filters(expr);
        let pushdown_op_filters = can_pushdown_op_filters(&expr.right);
        let handler = get_binary_op_handler(expr.op);
        Ok(Self {
            lhs,
            rhs,
            expr: expr.clone(),
            handler: Box::new( handler.deref() ),
            can_pushdown_filters,
            can_pushdown_op_filters: pushdown_op_filters
        })
    }

    pub fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }

    fn eval_args(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<(Vec<Timeseries>, Vec<Timeseries>)> {

        // todo: don't parallelize if both sides are scalar
        if !self.can_pushdown_filters {
            // Execute both sides in parallel, since it is impossible to push down common filters
            // from exprFirst to exprSecond.
            // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886

            // todo: we need to clone ec
            return match rayon::join(
                || self.lhs.eval(ctx, ec),
                || self.rhs.eval(ctx, ec),
            ) {
                (Ok(first), Ok(second)) => Ok((first, second)),
                (Err(err), _) => Err(err),
                (Ok(_), Err(err)) => Err(err)
            };
        }

        // Execute binary operation in the following way:
        //
        // 1) execute the expr_first
        // 2) get common label filters for series returned at step 1
        // 3) push down the found common label filters to expr_second. This filters out unneeded series
        //    during expr_second execution instead of spending compute resources on extracting and processing these series
        //    before they are dropped later when matching time series according to https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
        // 4) execute the expr_second with possible additional filters found at step 3
        //
        // Typical use cases:
        // - Kubernetes-related: show pod creation time with the node name:
        //
        //     kube_pod_created{namespace="prod"} * on (uid) group_left(node) kube_pod_info
        //
        //   Without the optimization `kube_pod_info` would select and spend compute resources
        //   for more time series than needed. The selected time series would be dropped later
        //   when matching time series on the right and left sides of binary operand.
        //
        // - Generic alerting queries, which rely on `info` metrics.
        //   See https://grafana.com/blog/2021/08/04/how-to-use-promql-joins-for-more-effective-queries-of-prometheus-metrics-at-scale/
        //
        // - Queries, which get additional labels from `info` metrics.
        //   See https://www.robustperception.io/exposing-the-software-version-to-prometheus
        let tss_first = self.lhs.eval(ctx, ec)?;

        let mut lfs = get_common_label_filters(&tss_first[0..]);
        trim_filters_by_group_modifier(&mut lfs, &self.expr);
        let sec = pushdown_binary_op_filters(&self.expr.right, &mut lfs);
        return match sec {
            Cow::Borrowed(_) => {
                // if it hasn't been modified, default to existing evaluator
                let tss_second = self.rhs.eval(ctx, ec)?;
                Ok((tss_first, tss_second))
            },
            Cow::Owned(expr) => {
                // todo !!!!: use parse cache
                let evaluator = create_evaluator(&expr)?;
                let tss_second = evaluator.eval(ctx, ec)?;
                Ok((tss_first, tss_second))
            }
        }
    }

}

impl Evaluator for BinaryOpEvaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &mut Context, ec: &EvalConfig) -> RuntimeResult<Vec<Timeseries>> {
        let mut tss_left: Vec<Timeseries> = vec![];
        let mut tss_right: Vec<Timeseries> = vec![];

        if self.expr.op == BinaryOp::And || self.expr.op == BinaryOp::If {
            // Fetch right-side series at first, since it usually contains
            // lower number of time series for `and` and `if` operator.
            // This should produce more specific label filters for the left side of the query.
            // This, in turn, should reduce the time to select series for the left side of the query.
            (tss_right, tss_left) = self.eval_args(ctx, ec)?
        } else {
            (tss_left, tss_right) = self.eval_args(ctx, ec)?
        }

        let mut bfa = BinaryOpFuncArg {
            be: self.expr.clone(),
            left: std::mem::take(&mut tss_left),
            right: std::mem::take(&mut tss_right),
        };

        match (self.handler)(&mut bfa) {
            Err(err) => Err(RuntimeError::from(
                format!("cannot evaluate {}: {:?}", &self.expr, err)
            )),
            Ok(v) => Ok(v)
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

fn is_aggr_func_without_grouping(e: &Expression) -> bool {
    match e {
        Expression::Aggregation(afe) => {
            if let Some(modifier) = &afe.modifier {
                modifier.args.len() == 0
            } else {
                true
            }
        },
        _ => false
    }
}

fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    let mut m: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for ts in tss.iter() {
        for (key, value) in ts.metric_name.iter() {
            if let Some(set) = m.get_mut(key) {
                set.insert(value.to_string());
            } else {
                let mut s = BTreeSet::new();
                s.insert(value.to_string());
                m.insert(key.to_string(), s);
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

        let mut vals: Vec<&String> = values.iter().collect::<Vec<_>>();
        vals.sort();

        let str_value: String;
        let mut is_regex = false;

        if values.len() == 1 {
            str_value = vals[0].into()
        } else {
            str_value = join_regexp_values(&vals);
            is_regex = true;
        }
        let lf = if is_regex {
            LabelFilter::equal(key, str_value).unwrap()
        } else {
            LabelFilter::regex_equal(key, str_value).unwrap()
        };

        lfs.push(lf);
    }
    lfs.sort_by(|a, b| a.label.cmp(&b.label));
    lfs
}


fn join_regexp_values(a: &Vec<&String>) -> String {
    let init_size = a.iter().fold(
        0,
        |res, x| res + x.len(),
    );
    let mut res = String::with_capacity(init_size);
    for (i, s) in a.iter().enumerate() {
        let s_quoted = escape(s);
        res.push_str(s_quoted.as_str());
        if i < a.len() - 1 {
            res.push('|')
        }
    }
    res
}
