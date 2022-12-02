use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use regex::escape;

use metricsql::ast::*;
use metricsql::functions::{DataType, Volatility};
use metricsql::transform::{
    pushdown_binary_op_filters
};

use crate::context::Context;
use crate::eval::{create_evaluator, eval_number, ExprEvaluator};
use crate::eval::binary_op::{BinaryOpFn, BinaryOpFuncArg, get_binary_op_handler};
use crate::eval::traits::Evaluator;
use crate::{EvalConfig, metric_name, Timeseries};
use crate::functions::types::AnyValue;
use crate::runtime_error::{RuntimeError, RuntimeResult};

use metric_name::Tag;
use metricsql::prelude::trim_filters_by_group_modifier;

pub struct BinaryEvaluator {
    expr: BinaryOpExpr,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    handler: Arc<dyn BinaryOpFn<Output=RuntimeResult<Vec<Timeseries>>>>,
    can_pushdown_filters: bool,
    can_parallelize: bool,
    return_type: DataType
}

impl BinaryEvaluator {
    pub fn new(expr: &BinaryOpExpr) -> RuntimeResult<Self> {
        let lhs = Box::new( create_evaluator(&expr.left)? );
        let rhs = Box::new(create_evaluator(&expr.right)? );
        let can_pushdown_filters = can_pushdown_common_filters(expr);
        let handler = get_binary_op_handler(expr.op);
        let rv = expr.return_value();
        let return_type = DataType::try_from(rv).unwrap_or(DataType::InstantVector);
        let can_parallelize = both_operands_are_vectors(&expr);

        Ok(Self {
            lhs,
            rhs,
            expr: expr.clone(),
            handler,
            can_pushdown_filters,
            can_parallelize,
            return_type
        })
    }

    pub fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }

    fn eval_args<'a>(&'a self, ctx: &Arc<&Context>, ec: &EvalConfig, swap: bool) -> RuntimeResult<(AnyValue, AnyValue)> {

        let (first, second, right_expr) = if swap {
            (&self.rhs, &self.lhs, &self.expr.right)
        } else {
            (&self.lhs, &self.rhs, &self.expr.left)
        };

        if !self.can_pushdown_filters {
            return if self.can_parallelize {
                // Execute both sides in parallel, since it is impossible to push down common filters
                // from exprFirst to exprSecond.
                // See https://github.com/VictoriaMetrics/VictoriaMetrics/issues/2886

                let mut ctx_clone = Arc::clone(ctx);
                match rayon::join(
                    || first.eval(ctx, ec),
                    || second.eval(&mut ctx_clone, ec),
                ) {
                    (Ok(first), Ok(second)) => Ok((first, second)),
                    (Err(err), _) => Err(err),
                    (Ok(_), Err(err)) => Err(err)
                }
            } else {
                // avoid multi-threading in simple case
                let left = first.eval(ctx, ec)?;
                let right = second.eval(ctx, ec)?;
                Ok((left, right))
            }

        }

        // Execute binary operation in the following way:
        //
        // 1) execute the expr_first
        // 2) get common label filters for series returned at step 1
        // 3) push down the found common label filters to expr_second. This filters out unneeded series
        //    during expr_second execution instead of spending compute resources on extracting and
        //    processing these series before they are dropped later when matching time series according to
        //    https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching
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
        //
        // Invariant: self.lhs and self.rhs are both DataType::InstantVector
        let mut first = first.eval(ctx, ec)?;
        // if first.is_empty() && self.op == Or, the result will be empty,
        // since the "exprFirst op exprSecond" would return an empty result in any case.
        // https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349
        if first.is_empty() && self.expr.op == BinaryOp::Or {
            return Ok((
                AnyValue::empty_vec(),
                AnyValue::empty_vec()
            ))
        }
        let sec = self.pushdown_filters(&mut first, &right_expr, &ec)?;
        return match sec {
            Cow::Borrowed(_) => {
                // if it hasn't been modified, default to existing evaluator
                let other = second.eval(ctx, ec)?;
                Ok((first, other))
            },
            Cow::Owned(expr) => {
                // todo(perf) !!!!: use ctx.parser_cache ?
                let evaluator = create_evaluator(&expr)?;
                let second = evaluator.eval(ctx, ec)?;
                Ok((first, second))
            }
        }
    }

    #[inline]
    fn pushdown_filters<'a>(&self, first: &mut AnyValue, dest: &'a Expression, ec: &EvalConfig) -> RuntimeResult<Cow<'a, Expression>> {
        let tss_first = first.as_instant_vec(ec)?;
        let mut lfs = get_common_label_filters(&tss_first[0..]);
        trim_filters_by_group_modifier(&mut lfs, &self.expr);
        let sec = pushdown_binary_op_filters(&dest, &mut lfs);
        Ok(sec)
    }

}

impl Evaluator for BinaryEvaluator {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &Arc<&Context>, ec: &EvalConfig) -> RuntimeResult<AnyValue> {
        // Determine if we should fetch right-side series at first, since it usually contains
        // lower number of time series for `and` and `if` operator.
        // This should produce more specific label filters for the left side of the query.
        // This, in turn, should reduce the time to select series for the left side of the query.
        let swap = self.expr.op == BinaryOp::And || self.expr.op == BinaryOp::If;

        let (left, right) = self.eval_args(ctx, ec, swap)?;
        let left_series = to_vector(ec, left)?;
        let right_series = to_vector(ec, right)?;

        let mut bfa = if swap {
            BinaryOpFuncArg::new(right_series,&self.expr, left_series)
        } else {
            BinaryOpFuncArg::new(left_series,&self.expr, right_series)
        };

        match (self.handler)(&mut bfa) {
            Err(err) => Err(RuntimeError::from(
                format!("cannot evaluate {}: {:?}", &self.expr, err)
            )),
            Ok(v) => Ok(AnyValue::InstantVector(v))
        }
    }

    fn return_type(&self) -> DataType {
        self.return_type
    }
}

fn to_vector(ec: &EvalConfig, value: AnyValue) -> RuntimeResult<Vec<Timeseries>> {
    match value {
        AnyValue::InstantVector(val) => Ok(val.into()), // todo: use std::mem::takee ??
        AnyValue::Scalar(n) => Ok(eval_number(ec, n)),
        _ => unreachable!("Bug: binary_op. Unexpected {} operand", value.data_type_name())
    }
}

fn both_operands_are_scalar(be: &BinaryOpExpr) -> bool {
    match (&be.left.return_value(), &be.right.return_value()) {
        (ReturnValue::Scalar, ReturnValue::Scalar) => true,
        _ => false
    }
}

fn both_operands_are_vectors(be: &BinaryOpExpr) -> bool {
    match (&be.left.return_value(), &be.right.return_value()) {
        (ReturnValue::InstantVector, ReturnValue::InstantVector) => true,
        _ => false
    }
}

fn can_pushdown_common_filters(be: &BinaryOpExpr) -> bool {
    if !both_operands_are_vectors(&be) {
        return false
    }
    match be.op {
        BinaryOp::Or | BinaryOp::Default => false,
        _=> {
            return !(
                is_aggr_func_without_grouping(&be.left) ||
                is_aggr_func_without_grouping(&be.right)
            );
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

pub(super) fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    let mut m: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for ts in tss.iter() {
        for Tag{ key: k, value: v} in ts.metric_name.tags.iter() {
            if let Some(set) = m.get_mut(k) {
                set.insert(v.to_string());
            } else {
                let mut s = BTreeSet::new();
                s.insert(v.to_string());
                m.insert(k.to_string(), s);
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
    // todo(perf): does this need to be sorted ?
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

#[cfg(test)]
mod tests {
    use metricsql::ast::MetricExpr;
    use crate::{Rows, Timeseries};
    use crate::eval::binary::get_common_label_filters;

    #[test]
    fn test_get_common_label_filters() {
        let f = |metrics: &str, lfs_expected: &str| {
            let mut tss: Vec<Timeseries> = vec![];

            let mut rows = Rows::try_from(metrics)
                .expect("error initializing rows from string");

            match rows.unmarshal(metrics) {
                Err(err) => {
                    panic!("unexpected error when parsing {}: {:?}", metrics, err);
                }
                Ok(_) => {}
            }
            for row in rows.iter() {
                let mut ts = Timeseries::default();
                for tag in row.tags.iter() {
                    ts.metric_name.set_tag(&tag.key, &tag.value);
                }
                tss.push(ts)
            }

            let lfs = get_common_label_filters(&tss);
            let me = MetricExpr::with_filters(lfs);

            let lfs_marshaled = me.to_string();
            assert_eq!(lfs_marshaled, lfs_expected,
                       "unexpected common label filters;\ngot\n{}\nwant\n{}", lfs_marshaled, lfs_expected)
        };

        f("", "{}");
        f("m 1", "{}");
        f(r#"m { a="b" } 1"#, r#"{a = "b"}"#);
        f(r#"m { c="d", a="b" } 1"#, r#"{a = "b", c = "d"}"#);
        f(r#"m1 { a="foo" } 1
          m2 { a="bar" } 1"#, r#"{a = ~"bar|foo"}"#);
        f(r#"m1 { a="foo" } 1
          m2 { b="bar" } 1"#, "{}");
        f(r#"m1 { a="foo", b="bar" } 1
          m2 { b="bar", c="x" } 1"#, r#"{b = "bar"}"#);
    }
}