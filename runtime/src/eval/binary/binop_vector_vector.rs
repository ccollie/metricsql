use std::borrow::Cow;
use std::collections::btree_set::BTreeSet;
use std::collections::HashMap;
use std::sync::Arc;

use regex::escape;
use tracing::{field, trace, trace_span, Span};

use metricsql::common::{LabelFilter, Operator};
use metricsql::prelude::*;

use crate::context::Context;
use crate::eval::binary::vector_binop_handlers::{exec_binop, BinaryOpFuncArg};
use crate::eval::exec::exec_expr;
use crate::eval::utils::series_len;
use crate::runtime_error::{RuntimeError, RuntimeResult};
use crate::types::Tag;
use crate::{EvalConfig, QueryValue, Timeseries};

pub(crate) fn eval_vector_vector_binop(
    expr: &BinaryExpr,
    ctx: &Arc<Context>,
    ec: &EvalConfig,
) -> RuntimeResult<QueryValue> {
    let is_tracing = ctx.trace_enabled();

    let span = if is_tracing {
        trace_span!("binary op", "op" = expr.op.as_str(), series = field::Empty)
    } else {
        Span::none()
    }
    .entered();

    // todo debug_assert!(expr.left.is_instant_vector() && expr.right.is_instant_vector());

    let (left, right) = if expr.op == Operator::And || expr.op == Operator::If {
        // Fetch right-side series at first, since it usually contains
        // lower number of time series for `and` and `if` operator.
        // This should produce more specific label filters for the left side of the query.
        // This, in turn, should reduce the time to select series for the left side of the query.
        exec_binary_op_args(ctx, ec, &expr.right, &expr.left, expr)?
    } else {
        exec_binary_op_args(ctx, ec, &expr.left, &expr.right, expr)?
    };

    let left_series = to_vector(left)?;
    let right_series = to_vector(right)?;
    let mut bfa = BinaryOpFuncArg::new(left_series, &expr, right_series);

    let result = exec_binop(&mut bfa)
        .map_err(|err| RuntimeError::from(format!("cannot evaluate {}: {:?}", &expr, err)))
        .and_then(|v| Ok(QueryValue::InstantVector(v)))?;

    if is_tracing {
        let series_count = series_len(&result);
        span.record("series", series_count);
    }

    Ok(result)
}

fn exec_binary_op_args(
    ctx: &Arc<Context>,
    ec: &EvalConfig,
    expr_first: &Expr,
    expr_second: &Expr,
    be: &BinaryExpr,
) -> RuntimeResult<(QueryValue, QueryValue)> {
    let can_push_down_filters = can_push_down_common_filters(be);

    if !can_push_down_filters {
        let op = be.op.as_str();
        let span = trace_span!("execute left and right sides in parallel", op);
        let _guard = span.enter();
        return match rayon::join(
            || {
                trace!("left");
                exec_expr(ctx, ec, expr_first)
            },
            || {
                trace!("right");
                let ctx_clone = Arc::clone(ctx);
                exec_expr(&ctx_clone, ec, expr_second)
            },
        ) {
            (Ok(first), Ok(second)) => Ok((first, second)),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
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
    let mut first = exec_expr(ctx, ec, expr_first)?;
    // if first.is_empty() && self.op == Or, the result will be empty,
    // since the "exprFirst op exprSecond" would return an empty result in any case.
    // https://github.com/VictoriaMetrics/VictoriaMetrics/issues/3349
    if first.is_empty() && be.op == Operator::Or {
        return Ok((QueryValue::empty_vec(), QueryValue::empty_vec()));
    }
    let sec_expr = push_down_filters(be, &mut first, &expr_second, &ec)?;
    let second = exec_expr(ctx, ec, &sec_expr)?;

    Ok((first, second))
}

fn push_down_filters<'a>(
    expr: &'a BinaryExpr,
    first: &mut QueryValue,
    dest: &'a Expr,
    ec: &EvalConfig,
) -> RuntimeResult<Cow<'a, Expr>> {
    let tss_first = first.as_instant_vec(ec)?;
    let mut common_filters = get_common_label_filters(&tss_first[0..]);
    if !common_filters.is_empty() {
        trim_filters_by_group_modifier(&mut common_filters, &expr);
        let mut copy = dest.clone();
        push_down_binary_op_filters_in_place(&mut copy, &mut common_filters);
        return Ok(Cow::Owned(copy));
    }
    Ok(Cow::Borrowed(dest))
}

fn to_vector(value: QueryValue) -> RuntimeResult<Vec<Timeseries>> {
    match value {
        QueryValue::InstantVector(val) => Ok(val.into()), // todo: use std::mem::take ??
        _ => unreachable!(
            "Bug: binary_op. Unexpected {} operand",
            value.data_type_name()
        ),
    }
}

fn can_push_down_common_filters(be: &BinaryExpr) -> bool {
    if matches!(be.op, Operator::Or | Operator::Default) {
        return false;
    }
    return match (&be.left.as_ref(), &be.right.as_ref()) {
        (Expr::Aggregation(left), Expr::Aggregation(right)) => {
            if left.is_non_grouping() || right.is_non_grouping() {
                return false;
            }
            true
        }
        _ => true,
    };
}

fn get_common_label_filters(tss: &[Timeseries]) -> Vec<LabelFilter> {
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
            // This may slow down the provider for matching time series.
            continue;
        }

        let vals: Vec<&String> = values.iter().collect::<Vec<_>>();

        let lf = if values.len() == 1 {
            LabelFilter::equal(key, vals[0].into()).unwrap()
        } else {
            let str_value = join_regexp_values(&vals);
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
