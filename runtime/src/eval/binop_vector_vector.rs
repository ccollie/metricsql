use std::borrow::Cow;
use std::collections::btree_set::BTreeSet;
use std::collections::HashMap;
use std::sync::Arc;

use regex::escape;
use tracing::{field, trace, trace_span, Span};

use metricsql::ast::*;
use metricsql::common::{LabelFilter, Operator, Value, ValueType};
use metricsql::functions::Volatility;
use metricsql::optimize::trim_filters_by_group_modifier;

use crate::context::Context;
use crate::eval::binop_handlers::{
    get_binary_op_func, BinaryOpFn, BinaryOpFuncArg, BinaryOpFuncResult,
};
use crate::eval::traits::Evaluator;
use crate::eval::{create_evaluator, eval_number, ExprEvaluator};
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::{EvalConfig, QueryValue, Timeseries};

use crate::eval::utils::series_len;
use crate::types::Tag;

pub struct BinaryEvaluatorVectorVector {
    expr: BinaryExpr,
    lhs: Box<ExprEvaluator>,
    rhs: Box<ExprEvaluator>,
    handler: Arc<dyn BinaryOpFn<Output = BinaryOpFuncResult>>,
    can_pushdown_filters: bool,
    can_parallelize: bool,
    /// Determine if we should fetch right-side series at first, since it usually contains
    /// lower number of time series for `and` and `if` operator.
    /// This should produce more specific label filters for the left side of the query.
    /// This, in turn, should reduce the time to select series for the left side of the query.
    swap: bool,
    return_type: ValueType,
}

impl BinaryEvaluatorVectorVector {
    pub fn new(expr: &BinaryExpr) -> RuntimeResult<Self> {
        let lhs = Box::new(create_evaluator(&expr.left)?);
        let rhs = Box::new(create_evaluator(&expr.right)?);
        let can_pushdown_filters = can_pushdown_common_filters(expr);
        let handler = get_binary_op_func(expr.op, expr.bool_modifier);
        let rv = expr.return_type();
        let return_type = ValueType::try_from(rv).unwrap_or(ValueType::InstantVector);
        let can_parallelize = should_parallelize(&expr);

        let swap = expr.op == Operator::And || expr.op == Operator::If;

        Ok(Self {
            lhs,
            rhs,
            expr: expr.clone(), // todo: store as arc on parse result and clone Arc
            handler,
            can_pushdown_filters,
            can_parallelize,
            return_type,
            swap,
        })
    }

    pub fn volatility(&self) -> Volatility {
        Volatility::Volatile
    }

    fn eval_args<'a>(
        &'a self,
        ctx: &Arc<Context>,
        ec: &EvalConfig,
    ) -> RuntimeResult<(QueryValue, QueryValue)> {
        let (first, second, right_expr) = if self.swap {
            (&self.rhs, &self.lhs, &self.expr.right)
        } else {
            (&self.lhs, &self.rhs, &self.expr.left)
        };

        if !self.can_pushdown_filters {
            // avoid multi-threading in simple case
            let op = self.expr.op.as_str();
            let span = trace_span!("execute left and right sides in parallel", op);
            let _guard = span.enter();
            return if self.can_parallelize {
                // todo: have a special path for case where both sides are selectors
                // i.e. both are io bound async
                match rayon::join(
                    || {
                        trace!("left");
                        first.eval(ctx, ec)
                    },
                    || {
                        trace!("right");
                        let ctx_clone = Arc::clone(ctx);
                        second.eval(&ctx_clone, ec)
                    },
                ) {
                    (Ok(first), Ok(second)) => Ok((first, second)),
                    (Err(err), _) => Err(err),
                    (Ok(_), Err(err)) => Err(err),
                }
            } else {
                let left = first.eval(ctx, ec)?;
                let right = second.eval(ctx, ec)?;
                Ok((left, right))
            };
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
        // Invariant: self.lhs and self.rhs are both ValueType::InstantVector
        let mut first = first.eval(ctx, ec)?;
        // if first.is_empty() && self.op == Or, the result will be empty,
        // since the "exprFirst op exprSecond" would return an empty result in any case.
        // https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349
        if first.is_empty() && self.expr.op == Operator::Or {
            return Ok((QueryValue::empty_vec(), QueryValue::empty_vec()));
        }
        let sec_expr = self.pushdown_filters(&mut first, &right_expr, &ec)?;
        return match sec_expr {
            Cow::Borrowed(_) => {
                // if it hasn't been modified, default to existing evaluator
                let other = second.eval(ctx, ec)?;
                Ok((first, other))
            }
            Cow::Owned(expr) => {
                // todo(perf) !!!!: use ctx.parser_cache ?
                let evaluator = create_evaluator(&expr)?;
                let second = evaluator.eval(ctx, ec)?;
                Ok((first, second))
            }
        };
    }

    fn pushdown_filters<'a>(
        &self,
        first: &mut QueryValue,
        dest: &'a Expr,
        ec: &EvalConfig,
    ) -> RuntimeResult<Cow<'a, Expr>> {
        if can_pushdown_filters(dest) {
            let tss_first = first.as_instant_vec(ec)?;
            let mut common_filters = get_common_label_filters(&tss_first[0..]);
            if !common_filters.is_empty() {
                trim_filters_by_group_modifier(&mut common_filters, &self.expr);
                let mut copy = dest.clone();
                push_down_binary_op_filters_in_place(&mut copy, &mut common_filters);
                return Ok(Cow::Owned(copy));
            }
        }
        Ok(Cow::Borrowed(dest))
    }
}

impl Value for BinaryEvaluatorVectorVector {
    fn value_type(&self) -> ValueType {
        self.return_type
    }
}

impl Evaluator for BinaryEvaluatorVectorVector {
    /// Evaluates and returns the result.
    fn eval(&self, ctx: &Arc<Context>, ec: &EvalConfig) -> RuntimeResult<QueryValue> {
        let is_tracing = ctx.trace_enabled();

        let span = if is_tracing {
            trace_span!(
                "binary op",
                "op" = self.expr.op.as_str(),
                series = field::Empty
            )
        } else {
            Span::none()
        }
        .entered();

        let (left, right) = self.eval_args(ctx, ec)?;
        let left_series = to_vector(ec, left)?;
        let right_series = to_vector(ec, right)?;

        let mut bfa = if self.swap {
            BinaryOpFuncArg::new(right_series, &self.expr, left_series)
        } else {
            BinaryOpFuncArg::new(left_series, &self.expr, right_series)
        };

        let result = match (self.handler)(&mut bfa) {
            Err(err) => Err(RuntimeError::from(format!(
                "cannot evaluate {}: {:?}",
                &self.expr, err
            ))),
            Ok(v) => Ok(QueryValue::InstantVector(v)),
        }?;

        if is_tracing {
            let series_count = series_len(&result);
            span.record("series", series_count);
        }

        Ok(result)
    }

    fn return_type(&self) -> ValueType {
        self.return_type
    }
}

fn to_vector(ec: &EvalConfig, value: QueryValue) -> RuntimeResult<Vec<Timeseries>> {
    match value {
        QueryValue::InstantVector(val) => Ok(val.into()), // todo: use std::mem::take ??
        QueryValue::Scalar(n) => Ok(eval_number(ec, n)),
        _ => unreachable!(
            "Bug: binary_op. Unexpected {} operand",
            value.data_type_name()
        ),
    }
}

fn should_parallelize_expr(expr: &BExpression) -> bool {
    use Expr::*;

    match expr.as_ref() {
        StringLiteral(_) | Number(_) | Duration(_) => false,
        _ => {
            // todo: maybe have a complexity threshold
            true
        }
    }
}

fn should_parallelize(be: &BinaryExpr) -> bool {
    if should_parallelize_expr(&be.left) && should_parallelize_expr(&be.right) {
        return true;
    }

    match (&be.left.return_type(), &be.right.return_type()) {
        (ValueType::InstantVector, ValueType::InstantVector) => true,
        _ => false,
    }
}

fn can_pushdown_common_filters(be: &BinaryExpr) -> bool {
    if !should_parallelize(&be) {
        return false;
    }
    match be.op {
        Operator::Or | Operator::Default => false,
        _ => {
            return !(is_aggr_func_without_grouping(&be.left)
                || is_aggr_func_without_grouping(&be.right));
        }
    }
}

fn is_aggr_func_without_grouping(e: &Expr) -> bool {
    match e {
        Expr::Aggregation(afe) => {
            if let Some(modifier) = &afe.modifier {
                modifier.args.len() == 0
            } else {
                true
            }
        }
        _ => false,
    }
}

pub(super) fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
    // todo(perf): use fnv or xxxhash
    let mut kv_map: HashMap<String, BTreeSet<String>> = HashMap::new();
    for ts in tss.iter() {
        for Tag { key: k, value: v } in ts.metric_name.tags.iter() {
            kv_map
                .entry(k.to_string())
                .or_insert_with(BTreeSet::new)
                .insert(v.to_string());
        }
    }

    let mut lfs: Vec<LabelFilter> = Vec::with_capacity(kv_map.len());
    for (key, values) in kv_map {
        if values.len() != tss.len() {
            // Skip the tag, since it doesn't belong to all the time series.
            continue;
        }

        if values.len() > 1000 {
            // Skip the filter on the given tag, since it needs to enumerate too many unique values.
            // This may slow down the search for matching time series.
            continue;
        }

        let vals: Vec<&String> = values.iter().collect::<Vec<_>>();

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
    lfs.sort();
    lfs
}

fn join_regexp_values(a: &Vec<&String>) -> String {
    let init_size = a.iter().fold(0, |res, x| res + x.len() + 3);
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

//#[cfg(test)]
/*
mod tests {
    use crate::eval::binary::get_common_label_filters;
    use crate::{Rows, Timeseries};
    use metricsql::ast::MetricExpr;

    #[test]
    fn test_get_common_label_filters() {
        let f = |metrics: &str, lfs_expected: &str| {
            let mut tss: Vec<Timeseries> = vec![];

            let mut rows = Rows::try_from(metrics).expect("error initializing rows from string");

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
            assert_eq!(
                lfs_marshaled, lfs_expected,
                "unexpected common label filters;\ngot\n{}\nwant\n{}",
                lfs_marshaled, lfs_expected
            )
        };

        f("", "{}");
        f("m 1", "{}");
        f(r#"m { a="b" } 1"#, r#"{a = "b"}"#);
        f(r#"m { c="d", a="b" } 1"#, r#"{a = "b", c = "d"}"#);
        f(
            r#"m1 { a="foo" } 1
          m2 { a="bar" } 1"#,
            r#"{a = ~"bar|foo"}"#,
        );
        f(
            r#"m1 { a="foo" } 1
          m2 { b="bar" } 1"#,
            "{}",
        );
        f(
            r#"m1 { a="foo", b="bar" } 1
          m2 { b="bar", c="x" } 1"#,
            r#"{b = "bar"}"#,
        );
    }
}


*/
