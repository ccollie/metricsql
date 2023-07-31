use std::borrow::Cow;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::vec::Vec;

use crate::ast::{AggregationExpr, BinaryExpr, Expr, RollupExpr};
use crate::common::{
    AggregateModifier, GroupModifierOp, JoinModifierOp, LabelFilter, Operator, NAME_LABEL,
};

/// push_down_filters optimizes e in order to improve its performance.
///
/// It performs the following optimizations:
///
/// - Adds missing filters to `foo{filters1} op bar{filters2}`
///   according to https://utcc.utoronto.ca/~cks/space/blog/sysadmin/PrometheusLabelNonOptimization
pub fn push_down_filters(expr: &Expr) -> Cow<Expr> {
    if can_pushdown_filters(expr) {
        let mut clone = expr.clone();
        optimize_label_filters_inplace(&mut clone);
        Cow::Owned(clone)
    } else {
        Cow::Borrowed(expr)
    }
}

pub fn can_pushdown_filters(expr: &Expr) -> bool {
    use Expr::*;

    match expr {
        Rollup(RollupExpr { expr, at, .. }) => {
            if let Some(at) = at {
                can_pushdown_filters(expr) || can_pushdown_filters(at)
            } else {
                can_pushdown_filters(expr)
            }
        }
        Function(f) => f.args.iter().any(can_pushdown_filters),
        Aggregation(agg) => agg.args.iter().any(can_pushdown_filters),
        BinaryOperator(_) => true,
        _ => false,
    }
}

pub(crate) fn optimize_label_filters_inplace(expr: &mut Expr) {
    use Expr::*;

    match expr {
        Rollup(re) => {
            optimize_label_filters_inplace(&mut re.expr);
            if let Some(ref mut at) = re.at {
                optimize_label_filters_inplace(at.as_mut());
            }
        }
        Function(f) => {
            for arg in f.args.iter_mut() {
                optimize_label_filters_inplace(arg);
            }
        }
        Aggregation(agg) => {
            for arg in agg.args.iter_mut() {
                optimize_label_filters_inplace(arg);
            }
        }
        BinaryOperator(be) => {
            optimize_label_filters_inplace(&mut be.left);
            optimize_label_filters_inplace(&mut be.right);
            let mut lfs = get_common_label_filters(expr);
            push_down_binary_op_filters_in_place(expr, &mut lfs);
        }
        _ => {}
    }
}

pub fn get_common_label_filters(e: &Expr) -> Vec<LabelFilter> {
    use Expr::*;

    match e {
        MetricExpression(m) => get_label_filters_without_metric_name(&m.label_filters),
        Rollup(r) => get_common_label_filters(&r.expr),
        Function(fe) => {
            if let Some(arg) = fe.get_arg_for_optimization() {
                get_common_label_filters(arg)
            } else {
                vec![]
            }
        }
        Aggregation(agg) => {
            if let Some(argument) = agg.get_arg_for_optimization() {
                let mut filters = get_common_label_filters(argument);
                trim_filters_by_aggr_modifier(&mut filters, agg);
                filters
            } else {
                vec![]
            }
        }
        BinaryOperator(e) => {
            let mut lfs_left = get_common_label_filters(&e.left);
            let mut lfs_right = get_common_label_filters(&e.right);
            match e.op {
                Operator::Or => {
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
                Operator::Unless => {
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

fn trim_filters_by_aggr_modifier(lfs: &mut Vec<LabelFilter>, afe: &AggregationExpr) {
    match &afe.modifier {
        None => lfs.clear(),
        Some(modifier) => match modifier {
            AggregateModifier::By(args) => filter_label_filters_on(lfs, args),
            AggregateModifier::Without(args) => filter_label_filters_ignoring(lfs, args),
        },
    }
}

/// trims lfs by the specified be.group_modifier.op (e.g. on() or ignoring()).
///
/// The following cases are possible:
/// - It returns lfs as is if be doesn't contain any group modifier
/// - It returns only filters specified in on()
/// - It drops filters specified inside ignoring()
pub fn trim_filters_by_group_modifier(lfs: &mut Vec<LabelFilter>, be: &BinaryExpr) {
    match &be.group_modifier {
        None => {}
        Some(modifier) => match modifier.op {
            GroupModifierOp::On => filter_label_filters_on(lfs, &modifier.labels),
            GroupModifierOp::Ignoring => filter_label_filters_ignoring(lfs, &modifier.labels),
        },
    }
}

// todo: use lifetimes instead of cloning
fn get_label_filters_without_metric_name(lfs: &[LabelFilter]) -> Vec<LabelFilter> {
    return lfs
        .iter()
        .filter(|x| x.label != NAME_LABEL)
        .cloned()
        .collect::<Vec<_>>();
}

/// Pushes down the given common_filters to e if possible.
///
/// e must be a part of binary operation - either left or right.
///
/// For example, if e contains `foo + sum(bar)` and common_filters=`{x="y"}`,
/// then the returned expression will contain `foo{x="y"} + sum(bar)`.
/// The `{x="y"}` cannot be pushed down to `sum(bar)`, since this
/// may change binary operation results.
pub fn pushdown_binary_op_filters<'a>(
    expr: &'a Expr,
    common_filters: &mut Vec<LabelFilter>,
) -> Cow<'a, Expr> {
    // according to pushdown_binary_op_filters_in_place, only the following types need to be
    // handled, so exit otherwise
    if common_filters.is_empty() || !can_pushdown_op_filters(expr) {
        return Cow::Borrowed(expr);
    }

    let mut copy = expr.clone();
    push_down_binary_op_filters_in_place(&mut copy, common_filters);
    Cow::Owned(copy)
}

pub fn can_pushdown_op_filters(expr: &Expr) -> bool {
    use Expr::*;
    // these are the types handled below in pushdown_binary_op_filters_in_place
    matches!(
        expr,
        MetricExpression(_) | Function(_) | Rollup(_) | BinaryOperator(_) | Aggregation(_)
    )
}

pub fn push_down_binary_op_filters_in_place(e: &mut Expr, common_filters: &mut Vec<LabelFilter>) {
    use Expr::*;

    if common_filters.is_empty() {
        return;
    }

    match e {
        MetricExpression(me) => {
            union_label_filters(&mut me.label_filters, common_filters);
            me.label_filters.sort();
        }
        Function(fe) => {
            if let Some(idx) = fe.arg_idx_for_optimization {
                let val = &fe.args[idx];
                // todo: check first if we can push down filters to the function
                // and only then do the actual pushdown (avoid a clone)
                let mut expr = val.clone();
                push_down_binary_op_filters_in_place(&mut expr, common_filters);
                fe.args[idx] = expr;
            }
        }
        BinaryOperator(bo) => {
            trim_filters_by_group_modifier(common_filters, bo);
            if !common_filters.is_empty() {
                push_down_binary_op_filters_in_place(&mut bo.left, common_filters);
                push_down_binary_op_filters_in_place(&mut bo.right, common_filters);
            }
        }
        Aggregation(aggr) => {
            trim_filters_by_aggr_modifier(common_filters, aggr);
            if !common_filters.is_empty() {
                if let Some(arg_idx) = aggr.arg_idx_for_optimization {
                    let expr = aggr.args.get_mut(arg_idx).unwrap();
                    push_down_binary_op_filters_in_place(expr, common_filters);
                }
            }
        }
        Rollup(re) => {
            push_down_binary_op_filters_in_place(&mut re.expr, common_filters);
        }
        _ => {}
    }
}

#[inline]
fn get_label_filters_map(filters: &[LabelFilter]) -> HashSet<String> {
    let set: HashSet<String> = HashSet::from_iter(filters.iter().map(|x| x.to_string()));
    set
}

fn intersect_label_filters(first: &mut Vec<LabelFilter>, second: &[LabelFilter]) {
    if first.is_empty() || second.is_empty() {
        return;
    }
    let set = get_label_filters_map(second);
    first.retain(|x| set.contains(&x.to_string()));
}

fn union_label_filters(a: &mut Vec<LabelFilter>, b: &Vec<LabelFilter>) {
    // todo (perf) do we need to clone, or can we drain ?
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
            // todo (perf): take from b, no alloc
            a.push(label.clone());
        }
    }
}

fn filter_label_filters_on(lfs: &mut Vec<LabelFilter>, args: &[String]) {
    if !args.is_empty() {
        let m: HashSet<&String> = HashSet::from_iter(args.iter());
        lfs.retain(|x| m.contains(&x.label))
    } else {
        lfs.clear()
    }
}

fn filter_label_filters_ignoring(lfs: &mut Vec<LabelFilter>, args: &[String]) {
    if !args.is_empty() {
        let m: HashSet<&String> = HashSet::from_iter(args.iter());
        lfs.retain(|x| !m.contains(&x.label))
    }
}
