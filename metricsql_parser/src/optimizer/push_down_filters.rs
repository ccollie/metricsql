use std::borrow::Cow;
use std::iter::FromIterator;
use std::vec::Vec;

use ahash::AHashSet;

use crate::ast::{
    AggregateModifier, AggregationExpr, Expr, Operator, RollupExpr, VectorMatchModifier,
};
use crate::label::{LabelFilter, NAME_LABEL};
use crate::prelude::VectorMatchCardinality;

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
        MetricExpression(m) => get_common_label_filters_without_metric_name(&m.label_filters),
        Rollup(r) => get_common_label_filters(&r.expr),
        Function(fe) => {
            if let Some(arg) = fe.arg_for_optimization() {
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
        UnaryOperator(unary) => get_common_label_filters(&unary.expr),
        BinaryOperator(binary) => {
            let mut lfs_left = get_common_label_filters(&binary.left);
            let mut lfs_right = get_common_label_filters(&binary.right);
            let card = VectorMatchCardinality::OneToOne;
            let group_modifier: Option<VectorMatchModifier> = None;

            let (group_modifier, join_modifier) = if let Some(modifier) = &binary.modifier {
                (&modifier.matching, &modifier.card)
            } else {
                (&group_modifier, &card)
            };

            match binary.op {
                Operator::Or => {
                    // {fCommon, f1} or {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on() {fCommon, f2} -> {}
                    // {fCommon, f1} or on(fCommon) {fCommon, f2} -> {fCommon}
                    // {fCommon, f1} or on(f1) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f2) {fCommon, f2} -> {}
                    // {fCommon, f1} or on(f3) {fCommon, f2} -> {}
                    intersect_label_filters(&mut lfs_left, &lfs_right);
                    trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                    lfs_left
                }
                Operator::Unless => {
                    // {f1} unless {f2} -> {f1}
                    // {f1} unless on() {f2} -> {}
                    // {f1} unless on(f1) {f2} -> {f1}
                    // {f1} unless on(f2) {f2} -> {}
                    // {f1} unless on(f1, f2) {f2} -> {f1}
                    // {f1} unless on(f3) {f2} -> {}
                    trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                    lfs_left
                }
                _ => {
                    match join_modifier {
                        // group_left
                        VectorMatchCardinality::ManyToOne(_) => {
                            // {f1} * group_left() {f2} -> {f1, f2}
                            // {f1} * on() group_left() {f2} -> {f1}
                            // {f1} * on(f1) group_left() {f2} -> {f1}
                            // {f1} * on(f2) group_left() {f2} -> {f1, f2}
                            // {f1} * on(f1, f2) group_left() {f2} -> {f1, f2}
                            // {f1} * on(f3) group_left() {f2} -> {f1}
                            trim_filters_by_match_modifier(&mut lfs_right, group_modifier);
                            union_label_filters(&mut lfs_left, &lfs_right);
                            lfs_left
                        }
                        // group_right
                        VectorMatchCardinality::OneToMany(_) => {
                            // {f1} * group_right() {f2} -> {f1, f2}
                            // {f1} * on() group_right() {f2} -> {f2}
                            // {f1} * on(f1) group_right() {f2} -> {f1, f2}
                            // {f1} * on(f2) group_right() {f2} -> {f2}
                            // {f1} * on(f1, f2) group_right() {f2} -> {f1, f2}
                            // {f1} * on(f3) group_right() {f2} -> {f2}
                            trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                            union_label_filters(&mut lfs_left, &lfs_right);
                            lfs_left
                        }
                        _ => {
                            // {f1} * {f2} -> {f1, f2}
                            // {f1} * on() {f2} -> {}
                            // {f1} * on(f1) {f2} -> {f1}
                            // {f1} * on(f2) {f2} -> {f2}
                            // {f1} * on(f1, f2) {f2} -> {f2}
                            // {f1} * on(f3} {f2} -> {}
                            union_label_filters(&mut lfs_left, &lfs_right);
                            trim_filters_by_match_modifier(&mut lfs_left, group_modifier);
                            lfs_left
                        }
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

/// trims lfs by the specified be.modifier.matching (e.g. on() or ignoring()).
///
/// The following cases are possible:
/// - It returns lfs as is if be doesn't contain any group modifier
/// - It returns only filters specified in on()
/// - It drops filters specified inside ignoring()
pub fn trim_filters_by_match_modifier(
    lfs: &mut Vec<LabelFilter>,
    group_modifier: &Option<VectorMatchModifier>,
) {
    match group_modifier {
        None => {}
        Some(modifier) => match modifier {
            VectorMatchModifier::On(labels) => filter_label_filters_on(lfs, labels.as_ref()),
            VectorMatchModifier::Ignoring(labels) => {
                filter_label_filters_ignoring(lfs, labels.as_ref())
            }
        },
    }
}

fn get_common_label_filters_without_metric_name(lfs: &Vec<LabelFilter>) -> Vec<LabelFilter> {
    if lfs.is_empty() {
        return vec![];
    }
    // let lfs_a = get_label_filters_without_metric_name(lfs);
    // for lfs in &lfss[1..].iter() {
    //     if lfs_a.is_empty() {
    //         return vec![];
    //     }
    //     let lfs_b = get_label_filters_without_metric_name(lfs);
    //     lfs_a = intersect_label_filters(lfs_a, lfs_b)
    // }
    get_label_filters_without_metric_name(lfs)
}

// todo: use lifetimes instead of cloning
fn get_label_filters_without_metric_name(lfs: &[LabelFilter]) -> Vec<LabelFilter> {
    return lfs
        .iter()
        .filter(|x| x.label != NAME_LABEL)
        .cloned()
        .collect::<Vec<_>>();
}

/// Pushes down the given common_filters to `expr` if possible.
///
/// `expr` must be a part of a binary operation - either left or right.
///
/// For example, if e contains `foo + sum(bar)` and common_filters=`{x="y"}`,
/// then the returned expression will contain `foo{x="y"} + sum(bar)`.
///
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
        MetricExpression(_)
            | Function(_)
            | Rollup(_)
            | BinaryOperator(_)
            | Aggregation(_)
            | UnaryOperator(_)
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
            if let Some(idx) = fe.arg_idx_for_optimization() {
                if let Some(val) = fe.args.get_mut(idx) {
                    push_down_binary_op_filters_in_place(val, common_filters);
                }
            }
        }
        UnaryOperator(unary) => {
            push_down_binary_op_filters_in_place(&mut unary.expr, common_filters);
        }
        BinaryOperator(bo) => {
            if let Some(modifier) = &bo.modifier {
                trim_filters_by_match_modifier(common_filters, &modifier.matching);
            }
            push_down_binary_op_filters_in_place(&mut bo.left, common_filters);
            push_down_binary_op_filters_in_place(&mut bo.right, common_filters);
        }
        Aggregation(aggr) => {
            trim_filters_by_aggr_modifier(common_filters, aggr);
            if let Some(arg_idx) = aggr.arg_idx_for_optimization() {
                if let Some(expr) = aggr.args.get_mut(arg_idx) {
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
fn get_label_filters_map(filters: &[LabelFilter]) -> AHashSet<String> {
    let set: AHashSet<String> = AHashSet::from_iter(filters.iter().map(|x| x.to_string()));
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
        let m: AHashSet<&String> = AHashSet::from_iter(args.iter());
        lfs.retain(|x| m.contains(&x.label))
    } else {
        lfs.clear()
    }
}

fn filter_label_filters_ignoring(lfs: &mut Vec<LabelFilter>, args: &[String]) {
    if !args.is_empty() {
        let m: AHashSet<&String> = AHashSet::from_iter(args.iter());
        lfs.retain(|x| !m.contains(&x.label))
    }
}
