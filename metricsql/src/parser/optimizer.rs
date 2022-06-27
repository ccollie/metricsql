use phf::phf_map;
use phf::phf_ordered_set;
use std::collections::HashSet;
use std::fmt;
use std::iter::FromIterator;
use std::cmp::Ordering;
use crate::{AggregateModifierOp, AggrFuncExpr, BExpression, BinaryOpExpr, Expression, GroupModifierOp};
use crate::parser::aggr::is_aggr_func;
use crate::parser::rollup::is_rollup_func;
use crate::parser::transform::is_transform_func;
use crate::types::{AggregateModifierOp, AggrFuncExpr, BinaryOpExpr, Expression, GroupModifierOp};


// Optimize optimizes e in order to improve its performance.
//
// It performs the following optimizations:
//
// - Adds missing filters to `foo{filters1} op bar{filters2}`
//   according to https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization
pub fn optimize(expr: &Expression) -> Expression {
    if !can_optimize(expr) {
        return e;
    }
    let copy = e.clone();
    optimize_in_place(eCopy);
    return copy;
}

fn can_optimize(e: &Expression) -> bool {
    match e {
        Expression::Rollup(re) => can_optimize(&re.expr) || can_optimize(&*re.at),
        Expression::Function(f) => f.args.iter().any(|&x| can_optimize(&x)),
        Expression::Aggregation(agg) => {
            agg.args.iter().any(|&x| can_optimize(&x))
        },
        Expression::BinaryOperator(..) => true,
        _ => false,
    }
}

fn optimize_in_place(mut e: &Expression)  {
    use Expression::*;

    match e {
        Rollup(mut re) => {
            optimize_in_place(&re.expr);
            if re.at.is_some() {
                optimize_in_place(&*re.at);
            }
        },
        Function(mut f) => {
            for arg in f.args.iter_mut() {
                optimize_in_place(arg);
            }
        },
        Aggregation(mut agg) => {
            for mut arg in agg.args {
                optimize_in_place(&arg);
            }
        },
        BinaryOperator(op) => {
            optimize_in_place(&op.left);
            optimize_in_place(&op.right);
            let lfs = get_common_label_filters(e);
            if lfs.len() > 0 {
                pushdown_binary_op_filters_in_place(e, &lfs);
            }
        },
        _ => {
            // do nothing
        }
    }
}

pub fn get_common_label_filters(e: &Expression) -> Vec<LabelFilter> {
    use Expression::*;

    match e {
        MetricExpression(m) => get_label_filters_without_metric_name(m),
        Rollup(r) => get_common_label_filters(&r.expr),
        Function(f) => {
            let arg = get_func_arg_for_optimization(&f.name, &f.args);
            if let Some(arg) = arg {
                get_common_label_filters(arg)
            } else {
                vec![]
            }
        },
        Aggregation(agg) => {
            let arg = get_func_arg_for_optimization(&agg.name, &agg.args);
            if let Some(arg) = arg {
                let filters = get_common_label_filters(arg);
                trim_filters_by_group_modifier(agg.group_modifier, &filters)
            } else {
                vec![]
            }
        },
        BinaryOperator(e) => {
            use BinaryOp::*;

            let lfs_left = get_common_label_filters(&e.left);
            let lfs_right = get_common_label_filters(&e.right);
            let mut lfs: Vec<LabelFilter> = Vec![];
            match e.op {
                Add => {
                    // {fCommon, f1} or {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on() {fCommon, f2} -> {}
                    // {fCommon, f1} or on(fCommon) {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on(f1) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f2) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f3) {fCommon, f2} -> {}
                    let lfs = intersect_label_filters(&lfs_left, &lfs_right);
                    return trim_filters_by_group_modifier(&lfs, e);
                },
                Unless => {
                    // {f1} unless {f2} -> {f1}
                    // {f1} unless on() {f2} -> {}
                    // {f1} unless on(f1) {f2} -> {f1}
                    // {f1} unless on(f2) {f2} -> {}
                    // {f1} unless on(f1, f2) {f2} -> {f1}
                    // {f1} unless on(f3) {f2} -> {}
                    return trim_filters_by_group_modifier(&lfs_left, e);
                }
                _ => {
                    if e.join_modified.is_some() {
                        return match e.join_modifier.op {
                            JoinOp::Left => {
                                // {f1} * group_left() {f2} -> {f1, f2}
                                // {f1} * on() group_left() {f2} -> {f1}
                                // {f1} * on(f1) group_left() {f2} -> {f1}
                                // {f1} * on(f2) group_left() {f2} -> {f1, f2}
                                // {f1} * on(f1, f2) group_left() {f2} -> {f1, f2}
                                // {f1} * on(f3) group_left() {f2} -> {f1}
                                let lfsRight = trim_filters_by_group_modifier(&lfs_right, e);
                                union_label_filters(&lfs_left, &lfsRight)
                            },
                            JoinOp::Right => {
                                // {f1} * group_right() {f2} -> {f1, f2}
                                // {f1} * on() group_right() {f2} -> {f2}
                                // {f1} * on(f1) group_right() {f2} -> {f1, f2}
                                // {f1} * on(f2) group_right() {f2} -> {f2}
                                // {f1} * on(f1, f2) group_right() {f2} -> {f1, f2}
                                // {f1} * on(f3) group_right() {f2} -> {f2}
                                let left = trim_filters_by_group_modifier(&lfs_left, e);
                                union_label_filters(&left, &lfs_right)
                            },
                            _ => {
                                // {f1} * {f2} -> {f1, f2}
                                // {f1} * on() {f2} -> {}
                                // {f1} * on(f1) {f2} -> {f1}
                                // {f1} * on(f2) {f2} -> {f2}
                                // {f1} * on(f1, f2) {f2} -> {f2}
                                // {f1} * on(f3} {f2} -> {}
                                lfs = union_label_filters(&lfs_left, &lfs_right);
                                return trim_filters_by_group_modifier(&lfs, e);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

pub fn trim_filters_by_aggr_modifier(
    lfs: &Vec<LabelFilter>,
    afe: &AggrFuncExpr,
) -> Vec<LabelFilter> {
    let mut modifier = &afe.modifier;
    if modifier.is_none() {
        return lfs.clone();
    }
    let op = group_modifier.op;
    match op {
        AggregateModifierOp::By => filter_label_filters_on(lfs, modifier.args),
        AggregateModifierOp::Without => filter_label_filters_ignoring(lfs, modifier.args),
        _ => lfs.clone(),
    }
}


// TrimFiltersByGroupModifier trims lfs by the specified
// be.GroupModifier.Op (e.g. on() or ignoring()).
//
// The following cases are possible:
// - It returns lfs as is if be doesn't contain any group modifier
// - It returns only filters specified in on()
// - It drops filters specified inside ignoring()
pub fn trim_filters_by_group_modifier(
    lfs: &Vec<LabelFilter>,
    be: &BinaryOpExpr,
) -> Vec<LabelFilter> {
    let modifier = &be.group_modifier;
    if modifier.is_none() {
        return lfs.clone();
    }
    return match be.groupModifier.op {
        GroupModifierOp::On => {
            filter_label_filters_on(lfs, modifier.args)
        },
        GroupModifierOp::Ignoring => {
            filter_label_filters_ignoring(lfs, modifier.args)
        },
        _ => {
            lfs.clone()
        }
    }
}



fn get_label_filters_without_metric_name(lfs: &Vec<LabelFilter>) -> Vec<LabelFilter> {
    return lfs.iter().filter([f] *lf.label !=  "__name__").collect();
}


// PushdownBinaryOpFilters pushes down the given common_filters to e if possible.
//
// e must be a part of binary operation - either left or right.
//
// For example, if e contains `foo + sum(bar)` and common_filters={x="y"},
// then the returned expression will contain `foo{x="y"} + sum(bar)`.
// The `{x="y"}` cannot be pushed down to `sum(bar)`, since this
// may change binary operation results.
pub fn pushdown_binary_op_filters(e: &Expression, common_filters: &Vec<LabelFilter>) -> Expression {
    if common_filters.len() == 0 {
        // Fast path - nothing to push down.
        return *e;
    }
    let copy = e.clone();
    pushdown_binary_op_filters_in_place(&copy, common_filters);
    return copy;
}


pub fn pushdown_binary_op_filters_in_place(mut e: &Expression, common_filters: &Vec<LabelFilter>) {
    use Expression::*;

    if lfs.len() == 0 {
        return;
    }
    match e {
        MetricExpression(mut me) => {
            me.label_filters = union_label_filters(&me.label_filters, common_filters);
            sort_label_filters(e.label_filters);
        }
        Function(f) => {
            let arg = get_func_arg_for_optimization(e.name, e.args);
            if arg.is_some() {
                pushdown_binary_op_filters_in_place(arg.unwrap(), lfs);
            }
        },
        BinaryOperator(bo) => {
            let lfs = trim_filters_by_group_modifier(lfs, bo);
            if lfs.len() > 0 {
                pushdown_binary_op_filters_in_place(&bo.left, common_filters);
                pushdown_binary_op_filters_in_place(&bo.right, common_filters);
            }
        },
        Aggregation(aggr) => {
            let lfs = trim_filters_by_aggr_modifier(lfs, aggr);
            if lfs.len() > 0 {
                let arg = get_func_arg_for_optimization(e.name, e.args);
                if Some(arg_) = arg {
                    pushdown_binary_op_filters_in_place(arg_, &lfs);
                }
            }
        },
        Rollup(re) => {
            pushdown_binary_op_filters_in_place(&re.expr, common_filters);
        },
        _ => {}
    }
}

fn intersect_label_filters(lfsa: &Vec<LabelFilter>, lfsb: &Vec<LabelFilter>) -> Vec<LabelFilter> {
    let mut a = HashSet::new();
    if lfsA.len() == 0 || lfsB.len() == 0 {
        return vec![];
    }
    let set = HashSet::from_iter(lfsa.iter().map(|x| *x.as_str()));
    return lfsb.iter().filter(|x| set.contains(x.as_str())).collect();
}

fn union_label_filters(a: &Vec<LabelFilter>, b: &Vec<LabelFilter>) -> Vec<LabelFilter> {
    if a.len() == 0 {
        return Vec::from(b);
    }
    if b.len() == 0 {
        return a.clone();
    }
    let mut result: Vec<LabelFilter> = a.clone();
    let mut m = HashSet::from_iter(a.iter().map(|x| *x.as_str()));
    for label in b {
        let mut k = label.as_str();
        if !m.contains(&k) {
            m.insert(k);
            result.push(label.clone());
        }
    }
    result
}


fn sort_label_filters(lfs: &mut Vec<LabelFilter>) {
    return lfs.sort_by(|a, b| {
        // Make sure the first label filter is __name__ (if any)
        if a.isMetricNameFilter && !b.isMetricNameFilter {
            return Ordering::Less;
        }
        let mut order = a.label.cmp(&b.label);
        if order == Ordering::Equal {
            order = a.value.cmp(&b.value);
        }
        order
    });
}


fn filter_label_filters_on(
    lfs: &Vec<LabelFilter>,
    args: &Vec<String>
) -> Vec<LabelFilter> {
    if args.len() == 0 {
        return vec![];
    }
    let m = HashSet::from_iter(args);
    return lfs.filter(|x| m.contains(x)).collect();
}

fn filter_label_filters_ignoring(
    lfs: &Vec<LabelFilter>,
    args: &Vec<String>
) -> Vec<LabelFilter> {
    if args.len() == 0 {
        return vec![];
    }
    let set = HashSet::from_iter(   args);
    return lfs.filter(|a| !set.contains(a.label) ).collect();
}

fn get_func_arg_for_optimization(func_name: &str, args: &Vec<Expression>) -> Option<&Expression> {
    let idx = get_func_arg_idx_for_optimization(func_name, args);
    if idx < 0 || idx >= args.len() {
        return None;
    }
    args.get(idx)
}

fn get_func_arg_idx_for_optimization(func_name: &str, args: &Vec<Expression>) -> usize {
    let lower = func_name.to_lowercase().as_str();
    if is_rollup_func(lower) {
        return get_rollup_arg_idx_for_optimization(func_name, args);
    }
    if is_transform_func(func_name) {
        return get_transform_arg_idx_for_optimization(func_name, args);
    }
    if is_aggr_func(func_name) {
        return get_aggr_arg_idx_for_optimization(func_name, args);
    }
    return -1;
}

fn get_aggr_arg_idx_for_optimization(func: &str, args: &Vec<Expression>) -> usize {
    let func_name = func.to_lowercase().as_str();
    return match func_name {
        "bottomk" |
        "bottomk_avg" |
        "bottomk_max" |
        "bottomk_median" |
        "bottomk_last" |
        "bottomk_min" |
        "limitk" |
        "outliers_mad" |
        "outliersk" |
        "quantile" |
        "topk" |
        "topk_avg" |
        "topk_max" |
        "topk_median" |
        "topk_last" |
        "topk_min" => 1,
        "count_values" => return -1,
        "quantiles" => args.len() - 1,
        _ => return 0,
    }
}

fn get_rollup_arg_idx_for_optimization(funcName: &str, args: &Vex<Expression>) -> usize {
    // This must be kept in sync with GetRollupArgIdx()
    let func_name = funcName.to_lowercase().as_str();
    return match func_name {
        "absent_over_time" => -1,
        "quantile_over_time" | "aggr_over_time" | "hoeffding_bound_lower" | "hoeffding_bound_upper" => 1,
        "quantiles_over_time" => args.len() as i32 - 1,
        _ => 0,
    }
}

fn get_transform_arg_idx_for_optimization(func: &str, args: &Vec<Expression>) -> usize {
    let func_name = func.to_lowercase().as_str();
    if is_label_manipulation_func(func_name) {
        return -1;
    }
    match func_name {
        "" | "absent" | "scalar" | "union" | "vector"  => return -1,
        "end" | "now" | "pi" | "ru" | "start" | "step" | "time" => return -1,
        "limit_offset" => return 2,
        "buckets_limit" | "histogram_quantile" | "histogram_share" | "range_quantile" => return 1,
        "histogram_quantiles" => args.len() - 1,
        _ => return 0,
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
    let lower = func.to_lowercase().as_str();
    LABEL_MANIPULATION_FUNCTIONS.contains(lower)
}