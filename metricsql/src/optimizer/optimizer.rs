use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::vec::Vec;

use phf::phf_ordered_set;

use crate::ast::*;
use crate::parser::aggr::is_aggr_func;
use crate::parser::rollup::is_rollup_func;
use crate::parser::transform::is_transform_func;

/// Optimize optimizes e in order to improve its performance.
///
/// It performs the following optimizations:
///
/// - Adds missing filters to `foo{filters1} op bar{filters2}`
///   according to https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization
pub fn optimize<'a>(expr: &'a Expression) -> Cow<'a, Expression> {
    if can_optimize(expr) {
        return optimize_internal(expr);
    }
    Cow::Borrowed::<'a>(expr)
}

pub fn can_optimize(e: &Expression) -> bool {
    match e {
        Expression::Rollup(re) => match (&re.expr, &re.at) {
            (expr, Some(at)) => can_optimize(expr) || can_optimize(at),
            _ => false,
        },
        Expression::Function(f) => f.args.iter().any(|x| can_optimize(x)),
        Expression::Aggregation(agg) => agg.args.iter().any(|x| can_optimize(x)),
        Expression::BinaryOperator(..) => true,
        _ => false,
    }
}

fn optimize_internal<'a>(e: &'a Expression) -> Cow<'a, Expression> {
    use Expression::*;

    match e {
        Rollup(re) => {
            let mut res = re.clone();
            if let Some(at) = res.at {
                res.at = Some(optimize_boxed(&at));
            }
            res.expr = optimize_boxed(&re.expr);
            Cow::Owned(Rollup(res))
        }
        Function(f) => {
            let mut res = f.clone();
            res.args = optimize_args(&f.args);
            Cow::Owned(Function(res))
        }
        Aggregation(agg) => {
            let mut res = agg.clone();
            res.args = optimize_args(&res.args);
            Cow::Owned(Aggregation(res))
        }
        BinaryOperator(be) => {
            let mut res = be.clone();
            res.left = optimize_boxed(&be.left);
            res.right = optimize_boxed(&be.right);
            let mut expr = BinaryOperator(res);
            let mut lfs = get_common_label_filters(&expr);
            if !lfs.is_empty() {
                pushdown_binary_op_filters_in_place(&mut expr, &mut lfs);
            }
            Cow::Owned(expr)
        }
        _ => Cow::Borrowed::<'a>(e)
    }
}

fn optimize_boxed(expr: &Expression) -> BExpression {
    let optimized = match optimize_internal(expr) {
        Cow::Owned(owned) => owned,
        Cow::Borrowed(borrowed) => borrowed.clone() // strange this compiles. should we clone ?
    };
    Box::new(optimized)
}

fn optimize_args(args: &[BExpression]) -> Vec<BExpression> {
    return args
        .iter()
        .map(|x| optimize_boxed(x))
        .collect::<_>();
}

pub fn get_common_label_filters(e: &Expression) -> Vec<LabelFilter> {
    use Expression::*;

    match e {
        MetricExpression(m) => get_label_filters_without_metric_name(&m.label_filters),
        Rollup(r) => get_common_label_filters(&r.expr),
        Function(f) => {
            let (_, arg) = get_func_arg_for_optimization(&f.name, &f.args);
            if let Some(arg) = arg {
                get_common_label_filters(arg)
            } else {
                vec![]
            }
        }
        Aggregation(agg) => {
            let (_, arg) = get_func_arg_for_optimization(&agg.name, &agg.args);
            if let Some(arg) = arg {
                let mut filters = get_common_label_filters(arg);
                trim_filters_by_aggr_modifier(&mut filters, agg);
                filters
            } else {
                vec![]
            }
        }
        BinaryOperator(e) => {
            use BinaryOp::*;

            let mut lfs_left = get_common_label_filters(&e.left);
            let mut lfs_right = get_common_label_filters(&e.right);
            match e.op {
                Or => {
                    // {fCommon, f1} or {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on() {fCommon, f2} -> {}
                    // {fCommon, f1} or on(fCommon) {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on(f1) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f2) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f3) {fCommon, f2} -> {}
                    intersect_label_filters(&mut lfs_left, &lfs_right);
                    trim_filters_by_group_modifier(&mut lfs_left, e);
                    lfs_left
                }
                Unless => {
                    // {f1} unless {f2} -> {f1}
                    // {f1} unless on() {f2} -> {}
                    // {f1} unless on(f1) {f2} -> {f1}
                    // {f1} unless on(f2) {f2} -> {}
                    // {f1} unless on(f1, f2) {f2} -> {f1}
                    // {f1} unless on(f3) {f2} -> {}
                    trim_filters_by_group_modifier(&mut lfs_left, e);
                    lfs_left
                }
                _ => {
                    if let Some(modifier) = &e.join_modifier {
                        match modifier.op {
                            JoinModifierOp::GroupLeft => {
                                // {f1} * group_left() {f2} -> {f1, f2}
                                // {f1} * on() group_left() {f2} -> {f1}
                                // {f1} * on(f1) group_left() {f2} -> {f1}
                                // {f1} * on(f2) group_left() {f2} -> {f1, f2}
                                // {f1} * on(f1, f2) group_left() {f2} -> {f1, f2}
                                // {f1} * on(f3) group_left() {f2} -> {f1}
                                trim_filters_by_group_modifier(&mut lfs_right, e);
                                union_label_filters(&mut lfs_left, &lfs_right);
                                lfs_left
                            }
                            JoinModifierOp::GroupRight => {
                                // {f1} * group_right() {f2} -> {f1, f2}
                                // {f1} * on() group_right() {f2} -> {f2}
                                // {f1} * on(f1) group_right() {f2} -> {f1, f2}
                                // {f1} * on(f2) group_right() {f2} -> {f2}
                                // {f1} * on(f1, f2) group_right() {f2} -> {f1, f2}
                                // {f1} * on(f3) group_right() {f2} -> {f2}
                                trim_filters_by_group_modifier(&mut lfs_left, e);
                                union_label_filters(&mut lfs_left, &lfs_right);
                                lfs_left
                            }
                        }
                    } else {
                        // {f1} * {f2} -> {f1, f2}
                        // {f1} * on() {f2} -> {}
                        // {f1} * on(f1) {f2} -> {f1}
                        // {f1} * on(f2) {f2} -> {f2}
                        // {f1} * on(f1, f2) {f2} -> {f2}
                        // {f1} * on(f3} {f2} -> {}
                        union_label_filters(&mut lfs_left, &lfs_right);
                        trim_filters_by_group_modifier(&mut lfs_left, e);
                        lfs_left
                    }
                }
            }
        }
        _ => {
            vec![]
        }
    }
}

pub fn trim_filters_by_aggr_modifier(lfs: &mut Vec<LabelFilter>, afe: &AggrFuncExpr) {
    match &afe.modifier {
        None => (),
        Some(modifier) => match modifier.op {
            AggregateModifierOp::By => filter_label_filters_on(lfs, &modifier.args),
            AggregateModifierOp::Without => filter_label_filters_ignoring(lfs, &modifier.args),
        },
    }
}

/// TrimFiltersByGroupModifier trims lfs by the specified
/// be.group_modifier.Op (e.g. on() or ignoring()).
///
/// The following cases are possible:
/// - It returns lfs as is if be doesn't contain any group modifier
/// - It returns only filters specified in on()
/// - It drops filters specified inside ignoring()
pub fn trim_filters_by_group_modifier(lfs: &mut Vec<LabelFilter>, be: &BinaryOpExpr) {
    match &be.group_modifier {
        None => {

        }
        Some(modifier) => match modifier.op {
            GroupModifierOp::On => filter_label_filters_on(lfs, &modifier.labels),
            GroupModifierOp::Ignoring => filter_label_filters_ignoring(lfs, &modifier.labels),
        },
    }
}

#[inline]
fn get_label_filters_without_metric_name(lfs: &[LabelFilter]) -> Vec<LabelFilter> {
    return lfs
        .iter()
        .filter(|x| x.label != "__name__")
        .cloned()
        .collect::<Vec<_>>();
}

/// Pushes down the given common_filters to e if possible.
///
/// e must be a part of binary operation - either left or right.
///
/// For example, if e contains `foo + sum(bar)` and common_filters={x="y"},
/// then the returned expression will contain `foo{x="y"} + sum(bar)`.
/// The `{x="y"}` cannot be pushed down to `sum(bar)`, since this
/// may change binary operation results.
pub fn pushdown_binary_op_filters<'a>(
    e: &'a Expression,
    common_filters: &mut Vec<LabelFilter>,
) -> Cow<'a, Expression> {
    // according to pushdown_binary_op_filters_in_place, only the following types need to be
    // handled, so exit otherwise
    if common_filters.is_empty() || !matches!(e, Expression::MetricExpression(_) |
        Expression::Function(_) |
        Expression::Rollup(_) |
        Expression::BinaryOperator(_) |
        Expression::Aggregation(_)) {
        return Cow::Borrowed(e)
    }

    let mut copy = e.clone();
    pushdown_binary_op_filters_in_place(&mut copy, common_filters);
    Cow::Owned(copy)
}

fn pushdown_binary_op_filters_in_place(e: &mut Expression, common_filters: &mut Vec<LabelFilter>) {
    use Expression::*;

    if common_filters.is_empty() {
        return;
    }
    match e {
        MetricExpression(me) => {
            union_label_filters(&mut me.label_filters, common_filters);
            sort_label_filters(&mut me.label_filters);
        }
        Function(fe) => {
            let (idx, arg) = get_func_arg_for_optimization(&fe.name, &fe.args);
            if let Some(val) = arg {
                let mut expr = val.clone();
                pushdown_binary_op_filters_in_place(&mut expr, common_filters);
                fe.args[idx] = expr;
            }
        }
        BinaryOperator(bo) => {
            trim_filters_by_group_modifier(common_filters, bo);
            if !common_filters.is_empty() {
                pushdown_binary_op_filters_in_place(&mut bo.left, common_filters);
                pushdown_binary_op_filters_in_place(&mut bo.right, common_filters);
            }
        }
        Aggregation(aggr) => {
            trim_filters_by_aggr_modifier(common_filters, aggr);
            if !common_filters.is_empty() {
                let (idx, arg) = get_func_arg_for_optimization(&aggr.name, &aggr.args);
                if let Some(argument) = arg {
                    let mut expr = argument.clone();
                    pushdown_binary_op_filters_in_place(&mut expr, common_filters);
                    aggr.args[idx] = expr;
                }
            }
        }
        Rollup(re) => {
            pushdown_binary_op_filters_in_place(&mut re.expr, common_filters);
        }
        _ => {}
    }
}

#[inline]
fn get_label_filters_map(filters: &[LabelFilter]) -> HashSet<String> {
    let set: HashSet<String> = HashSet::from_iter(filters.iter().map(|x| x.to_string()));
    set
}

pub fn intersect_label_filters(first: &mut Vec<LabelFilter>, second: &[LabelFilter]) {
    if first.is_empty() || second.is_empty() {
        return
    }
    let set = get_label_filters_map(second);
    first.retain(|x| set.contains(&x.to_string()));
}

pub fn union_label_filters(a: &mut Vec<LabelFilter>, b: &Vec<LabelFilter>) {
    //todo (perf) do we need to clone, or can we drain ?
    if a.is_empty() {
        a.append(&mut b.clone());
        return;
    }
    if b.is_empty() {
        return;
    }
    let m = get_label_filters_map(a);
    for label in b.iter() {
        let k = label.to_string();
        if !m.contains(&k) {
            a.push(label.clone());
        }
    }
}

fn sort_label_filters(lfs: &mut [LabelFilter]) {
    lfs.sort_by(|a, b| {
        // Make sure the first label filter is __name__ (if any)
        if a.is_metric_name_filter() && !b.is_metric_name_filter() {
            return Ordering::Less;
        }
        let mut order = a.label.cmp(&b.label);
        if order == Ordering::Equal {
            order = a.value.cmp(&b.value);
        }
        order
    })
}

fn filter_label_filters_on(lfs: &mut Vec<LabelFilter>, args: &[String]) {
    if !args.is_empty() {
        let m: HashSet<&String> = HashSet::from_iter(args.iter());
        lfs.retain(|x| m.contains(&x.label))
    }
}

fn filter_label_filters_ignoring(lfs: &mut Vec<LabelFilter>, args: &[String]) {
    if !args.is_empty() {
        let m: HashSet<&String> = HashSet::from_iter(args.iter());
        lfs.retain(|x| !m.contains(&x.label))
    }
}

fn get_func_arg_for_optimization<'a>(
    func_name: &str,
    args: &'a [BExpression],
) -> (usize, Option<&'a BExpression>) {
    let idx = get_func_arg_idx_for_optimization(func_name, args);
    if idx < 0 {
        (usize::MAX, None)
    } else {
        (idx as usize, args.get(idx as usize))
    }
}

fn get_func_arg_idx_for_optimization(func_name: &str, args: &[BExpression]) -> i16 {
    let lower = func_name.to_lowercase();
    if is_rollup_func(&lower) {
        return get_rollup_arg_idx_for_optimization(func_name, args) as i16;
    }
    if is_transform_func(func_name) {
        return get_transform_arg_idx_for_optimization(func_name, args) as i16;
    }
    if is_aggr_func(func_name) {
        return get_aggr_arg_idx_for_optimization(func_name, args) as i16;
    }
    -1
}

fn get_aggr_arg_idx_for_optimization(func: &str, args: &[BExpression]) -> i16 {
    let func_name = func.to_lowercase();
    return match func_name.as_str() {
        "bottomk" | "bottomk_avg" | "bottomk_max" | "bottomk_median" | "bottomk_last"
        | "bottomk_min" | "limitk" | "outliers_mad" | "outliersk" | "quantile" | "topk"
        | "topk_avg" | "topk_max" | "topk_median" | "topk_last" | "topk_min" => 1,
        "count_values" => return -1,
        "quantiles" => (args.len() - 1) as i16,
        _ => return 0,
    };
}

fn get_rollup_arg_idx_for_optimization(func_name: &str, args: &[BExpression]) -> i16 {
    // This must be kept in sync with GetRollupArgIdx()
    let lower = func_name.to_lowercase();
    return match lower.as_str() {
        "absent_over_time" => -1,
        "quantile_over_time"
        | "aggr_over_time"
        | "hoeffding_bound_lower"
        | "hoeffding_bound_upper" => 1,
        "quantiles_over_time" => (args.len() - 1) as i16,
        _ => 0,
    };
}

fn get_transform_arg_idx_for_optimization(func: &str, args: &[BExpression]) -> i16 {
    let func_name = func.to_lowercase();
    if is_label_manipulation_func(&func_name) {
        return -1;
    }
    match func_name.as_str() {
        "" | "absent" | "scalar" | "union" | "vector" => -1,
        "end" | "now" | "pi" | "ru" | "start" | "step" | "time" => -1,
        "limit_offset" => 2,
        "buckets_limit" | "histogram_quantile" | "histogram_share" | "range_quantile" => 1,
        "histogram_quantiles" => (args.len() - 1) as i16,
        _ => 0,
    }
}

static LABEL_MANIPULATION_FUNCTIONS: phf::OrderedSet<&'static str> = phf_ordered_set! {
    "alias",
   "drop_common_labels",
   "label_copy",
   "label_del",
   "label_graphite_group",
   "label_join",
   "label_keep",
   "label_lowercase",
   "label_map",
   "label_match",
   "label_mismatch",
   "label_move",
   "label_replace",
   "label_set",
   "label_transform",
   "label_uppercase",
   "label_value",
};

fn is_label_manipulation_func(func: &str) -> bool {
    let lower = func.to_lowercase();
    LABEL_MANIPULATION_FUNCTIONS.contains(&lower)
}
