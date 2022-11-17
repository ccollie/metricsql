use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::vec::Vec;

use crate::ast::*;

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

/// trims lfs by the specified be.group_modifier.op (e.g. on() or ignoring()).
///
/// The following cases are possible:
/// - It returns lfs as is if be doesn't contain any group modifier
/// - It returns only filters specified in on()
/// - It drops filters specified inside ignoring()
pub fn trim_filters_by_group_modifier(lfs: &mut Vec<LabelFilter>, be: &BinaryOpExpr) {
    match &be.group_modifier {
        None => {}
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
/// For example, if e contains `foo + sum(bar)` and common_filters=`{x="y"}`,
/// then the returned expression will contain `foo{x="y"} + sum(bar)`.
/// The `{x="y"}` cannot be pushed down to `sum(bar)`, since this
/// may change binary operation results.
pub fn pushdown_binary_op_filters<'a>(
    e: &'a Expression,
    common_filters: &mut Vec<LabelFilter>,
) -> Cow<'a, Expression> {
    // according to pushdown_binary_op_filters_in_place, only the following types need to be
    // handled, so exit otherwise
    if common_filters.is_empty() || !can_pushdown_op_filters(e){
        return Cow::Borrowed(e)
    }

    let mut copy = e.clone();
    pushdown_binary_op_filters_in_place(&mut copy, common_filters);
    Cow::Owned(copy)
}

#[inline]
pub fn can_pushdown_op_filters(e: &Expression) -> bool {
    // these are the types handled below in pushdown_binary_op_filters_in_place
    matches!(e, Expression::MetricExpression(_) |
        Expression::Function(_) |
        Expression::Rollup(_) |
        Expression::BinaryOperator(_) |
        Expression::Aggregation(_))
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
            if let Some(idx) = fe.get_arg_idx_for_optimization() {
                let val = &fe.args[idx];
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
                let arg = aggr.get_arg_for_optimization();
                if let Some(argument) = arg {
                    let mut expr = argument.clone();
                    pushdown_binary_op_filters_in_place(&mut expr, common_filters);
                    // .unwrap is safe here since get_arg_idx_for_optimization is used internally by
                    // get_arg_for_optimization
                    let idx = aggr.get_arg_idx_for_optimization().unwrap();
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


#[cfg(test)]
mod tests {
    use crate::ast::{Expression, MetricExpr};
    use crate::optimizer::{get_common_label_filters, optimize, pushdown_binary_op_filters};
    use crate::parser::parse;
    use test_case::test_case;

    #[test_case("{}", "{}")]
    #[test_case("foo", "{}")]
    #[test_case(r#"{__name__="foo"}"#, "{}")]
    #[test_case(r#"{__name__=~"bar"}"#, "{}")]
    #[test_case(r#"{__name__=~"a|b",x="y"}"#, r#"{x="y"}"#)]
    #[test_case(r#"foo{c!="d",a="b"}"#, r#"{c!="d", a="b"}"#)]
    #[test_case("1+foo", "{}")]
    #[test_case(r#"foo + bar{a="b"}"#, r#"{a="b"}"#)]
    #[test_case(r#"foo + bar / baz{a="b"}"#, r#"{a="b"}"#)]
    #[test_case(r#"foo{x!="y"} + bar / baz{a="b"}"#, r#"{x!="y", a="b"}"#)]
    #[test_case(r#"foo{x!="y"} + bar{x=~"a|b",q!~"we|rt"} / baz{a="b"}"#, r#"{x!="y", x=~"a|b", q!~"we|rt", a="b"}"#)]
    #[test_case(r#"{a="b"} + on() {c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b"} + on() group_left() {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} + on(a) group_left() {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} + on(c) group_left() {c="d"}"#, r#"{a="b", c="d"}"#)]
    #[test_case(r#"{a="b"} + on(a,c) group_left() {c="d"}"#, r#"{a="b", c="d"}"#)]
    #[test_case(r#"{a="b"} + on(d) group_left() {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} + on() group_right(s) {c="d"}"#, r#"{c="d"}"#)]
    #[test_case(r#"{a="b"} + On(a) groUp_right() {c="d"}"#, r#"{a="b", c="d"}"#)]
    #[test_case(r#"{a="b"} + on(c) group_right() {c="d"}"#, r#"{c="d"}"#)]
    #[test_case(r#"{a="b"} + on(a,c) group_right() {c="d"}"#, r#"{a="b", c="d"}"#)]
    #[test_case(r#"{a="b"} + on(d) group_right() {c="d"}"#, r#"{c="d"}"#)]
    #[test_case(r#"{a="b"} or {c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b",x="y"} or {x="y",c="d"}"#, r#"{x="y"}"#)]
    #[test_case(r#"{a="b",x="y"} Or on() {x="y",c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b",x="y"} Or on(a) {x="y",c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b",x="y"} Or on(x) {x="y",c="d"}"#, r#"{x="y"}"#)]
    #[test_case(r#"{a="b",x="y"} Or oN(x,y) {x="y",c="d"}"#, r#"{x="y"}"#)]
    #[test_case(r#"{a="b",x="y"} Or on(y) {x="y",c="d"}"#, r#"{}"#)]
    #[test_case(r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} unless {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} unless on() {c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b"} unLess on(a) {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} unLEss on(c) {c="d"}"#, r#"{}"#)]
    #[test_case(r#"{a="b"} unless on(a,c) {c="d"}"#, r#"{a="b"}"#)]
    #[test_case(r#"{a="b"} Unless on(x) {c="d"}"#, r#"{}"#)]
    fn test_get_common_label_filters(q: &str, result_expected: &str) {
        let e = parse(q).expect(format!("Error parsing expression: {}", q).as_str());
        let lfs = get_common_label_filters(&e);
        let me = MetricExpr::with_filters(lfs);
        let result = me.to_string();
        assert_eq!(&result, result_expected,
            "unexpected result for get_common_label_filters({});\ngot\n{}\nwant\n{}", q, result, result_expected);
    }

    #[test_case("foo", "{}","foo")]
    #[test_case("foo", r#"{a="b"}"#, r#"foo{a="b"}"#)]
    #[test_case(r#"foo + bar{x="y"}"#, r#"{c="d",a="b"}"#, r#"foo{a="b", c="d"} + bar{a="b", c="d", x="y"}"#)]
    #[test_case("sum(x)", r#"{a="b"}"#, "sum(x)")]
    #[test_case("foo or bar", r#"{a="b"}"#, r#"foo{a="b"} or bar{a="b"}"#)]
    #[test_case("foo or on(x) bar", r#"{a="b"}"#, r#"foo or on (x) bar"#)]
    #[test_case("foo == on(x) group_LEft bar", r#"{a="b"}"#, "foo == on (x) group_left () bar")]
    #[test_case(r#"foo{x="y"} > ignoRIng(x) group_left(abc) bar"#, r#"{a="b"}"#, r#"foo{a="b", x="y"} > ignoring (x) group_left (abc) bar{a="b"}"#)]
    #[test_case(r#"foo{x="y"} >bool ignoring(x) group_right(abc,def) bar"#, r#"{a="b"}"#, r#"foo{a="b", x="y"} > bool ignoring (x) group_right (abc, def) bar{a="b"}"#)]
    #[test_case("foo * ignoring(x) bar", r#"{a="b"}"#, r#"foo{a="b"} * ignoring (x) bar{a="b"}"#)]
    #[test_case(r#"foo{f1!~"x"} UNLEss bar{f2=~"y.+"}"#, r#"{a="b",x=~"y"}"#,
    r#"foo{a="b", f1!~"x", x=~"y"} unless bar{a="b", f2=~"y.+", x=~"y"}"#)]
    #[test_case("a / sum(x)", r#"{a="b",c=~"foo|bar"}"#, r#"a{a="b", c=~"foo|bar"} / sum(x)"#)]
    #[test_case(r#"round(rate(x[5m] offset -1h)) + 123 / {a="b"}"#, r#"{x!="y"}"#,
    r#"round(rate(x{x!="y"}[5m] offset -1h)) + (123 / {a="b", x!="y"})"#)]
    #[test_case("scalar(foo)+bar", r#"{a="b"}"#, r#"scalar(foo) + bar{a="b"}"#)]
    #[test_case("vector(foo)", r#"{a="b"}"#, "vector(foo)")]
    #[test_case(r#"{a="b"} + on() group_left() {c="d"}"#, r#"{a="b"}"#, r#"{a="b"} + on () group_left () {c="d"}"#)]
    fn test_pushdown_binary_op_filters(q: &str, filters: &str, result_expected: &str) {
        let e = parse(q).unwrap();
        let s_orig = format!("{}", e);
        let filters_exprs = parse(filters).unwrap_or_else(|_|
            panic!("cannot parse filters {}", filters)
        );
        match filters_exprs {
            Expression::MetricExpression(mut me) => {
                let result_expr = pushdown_binary_op_filters(&e, &mut me.label_filters);
                let result = format!("{}", result_expr);
                assert_eq!(result, result_expected,
                           "unexpected result for pushdown_binary_op_filters({}, {});\ngot\n{}\nwant\n{}", q, filters, result, result_expected);
                // Verify that the original e didn't change after pushdown_binary_op_filters() call
                let s = e.to_string();
                assert_eq!(s, s_orig, "the original expression has been changed;\ngot\n{}\nwant\n{}", s, s_orig)
            },
            _ => {
                panic!("filters={} must be a metrics expression; got {}", filters, filters_exprs)
            }
        }
    }

    fn test_optimize(q: &str, expected: &str) {
        let e = parse(q).expect(format!("Error parsing expression: {}", q).as_str());
        let s_orig = e.to_string();
        let e_optimized = optimize(&e);
        let q_optimized = e_optimized.to_string();
        assert_eq!(q_optimized, expected,
                   "unexpected q_optimized;\ngot\n{}\nwant\n{}", q_optimized, expected);
        // Make sure the the original e didn't change after Optimize() call
        let binding = e.to_string();
        let s = binding.as_str();
        assert_eq!(s, s_orig, 
            "the original expression has been changed;\ngot\n{}\nwant\n{}", s, s_orig)
    }

    // common binary expressions
    #[test_case("a + b", "a + b")]
    #[test_case(r#"foo{label1="value1"} == bar"#, r#"foo{label1="value1"} == bar{label1="value1"}"#)]
    #[test_case(r#"foo{label1="value1"} == bar{label2="value2"}"#, r#"foo{label1="value1", label2="value2"} == bar{label1="value1", label2="value2"}"#)]
    #[test_case(r#"foo + bar{b=~"a.*", a!="ss"}"#, r#"foo{a!="ss", b=~"a.*"} + bar{a!="ss", b=~"a.*"}"#)]
    #[test_case(r#"foo{bar="1"} / 234"#, r#"foo{bar="1"} / 234"#)]
    #[test_case(r#"foo{bar="1"} / foo{bar="1"}"#, r#"foo{bar="1"} / foo{bar="1"}"#)]
    #[test_case(r#"123 + foo{bar!~"xx"}"#, r#"123 + foo{bar!~"xx"}"#)]
    #[test_case(r#"foo or bar{x="y"}"#, r#"foo or bar{x="y"}"#)]
    #[test_case(r#"foo{x="y"} * on() baz{a="b"}"#, r#"foo{x="y"} * on () baz{a="b"}"#)]
    #[test_case(r#"foo{x="y"} * on(a) baz{a="b"}"#, r#"foo{a="b", x="y"} * on (a) baz{a="b"}"#)]
    #[test_case(r#"foo{x="y"} * on(bar) baz{a="b"}"#, r#"foo{x="y"} * on (bar) baz{a="b"}"#)]
    #[test_case(r#"foo{x="y"} * on(x,a,bar) baz{a="b"}"#, r#"foo{a="b", x="y"} * on (x, a, bar) baz{a="b", x="y"}"#)]
    #[test_case(r#"foo{x="y"} * ignoring() baz{a="b"}"#, r#"foo{a="b", x="y"} * ignoring () baz{a="b", x="y"}"#)]
    #[test_case(r#"foo{x="y"} * ignoring(a) baz{a="b"}"#, r#"foo{x="y"} * ignoring (a) baz{a="b", x="y"}"#)]
    #[test_case(r#"foo{x="y"} * ignoring(bar) baz{a="b"}"#, r#"foo{a="b", x="y"} * ignoring (bar) baz{a="b", x="y"}"#)]
    #[test_case(r#"foo{x="y"} * ignoring(x,a,bar) baz{a="b"}"#, r#"foo{x="y"} * ignoring (x, a, bar) baz{a="b"}"#)]
    #[test_case(r#"foo{x="y"} * ignoring() group_left(foo,bar) baz{a="b"}"#, r#"foo{a="b", x="y"} * ignoring () group_left (foo, bar) baz{a="b", x="y"}"#)]
    #[test_case(r#"foo{x="y"} * on(a) group_left baz{a="b"}"#, r#"foo{a="b", x="y"} * on (a) group_left () baz{a="b"}"#)]
    #[test_case(r#"foo{x="y"} * on(a) group_right(x, y) baz{a="b"}"#, r#"foo{a="b", x="y"} * on (a) group_right (x, y) baz{a="b"}"#)]
    #[test_case(r#"f(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#, r#"f(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#)]
    #[test_case(r#"sum(foo, bar{baz=~"sdf"} + aa{baz=~"axx", aa="b"})"#, r#"sum(foo, bar{aa="b", baz=~"axx", baz=~"sdf"} + aa{aa="b", baz=~"axx", baz=~"sdf"})"#)]
    #[test_case(r#"foo AND bar{baz="aa"}"#, r#"foo{baz="aa"} and bar{baz="aa"}"#)]
    #[test_case(r#"{x="y",__name__="a"} + {a="b"}"#, r#"a{a="b", x="y"} + {a="b", x="y"}"#)]
    #[test_case(r#"{x="y",__name__=~"a|b"} + {a="b"}"#, r#"{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#)]
    #[test_case(r#"a{x="y",__name__=~"a|b"} + {a="b"}"#, r#"a{__name__=~"a|b", a="b", x="y"} + {a="b", x="y"}"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on() group_left() {e="f"})"#, r#"{a="b", c="d"} + ({c="d"} * on () group_left () {e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(a) group_left() {e="f"})"#, r#"{a="b", c="d"} + ({a="b", c="d"} * on (a) group_left () {a="b", e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(c) group_left() {e="f"})"#, r#"{a="b", c="d"} + ({c="d"} * on (c) group_left () {c="d", e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(e) group_left() {e="f"})"#, r#"{a="b", c="d", e="f"} + ({c="d", e="f"} * on (e) group_left () {e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(x) group_left() {e="f"})"#, r#"{a="b", c="d"} + ({c="d"} * on (x) group_left () {e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on() group_right() {e="f"})"#, r#"{a="b", e="f"} + ({c="d"} * on () group_right () {e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(a) group_right() {e="f"})"#, r#"{a="b", e="f"} + ({a="b", c="d"} * on (a) group_right () {a="b", e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(c) group_right() {e="f"})"#, r#"{a="b", c="d", e="f"} + ({c="d"} * on (c) group_right () {c="d", e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(e) group_right() {e="f"})"#, r#"{a="b", e="f"} + ({c="d", e="f"} * on (e) group_right () {e="f"})"#)]
    #[test_case(r#"{a="b"} + ({c="d"} * on(x) group_right() {e="f"})"#, r#"{a="b", e="f"} + ({c="d"} * on (x) group_right () {e="f"})"#)]
    fn common_binary_expressions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }

    // specially handled binary expressions
    #[test_case(r#"foo{a="b"} or bar{x="y"}"#, r#"foo{a="b"} or bar{x="y"}"#)]
    #[test_case(r#"(foo{a="b"} + bar{c="d"}) or (baz{x="y"} <= x{a="b"})"#, r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or (baz{a="b", x="y"} <= x{a="b", x="y"})"#)]
    #[test_case(r#"(foo{a="b"} + bar{c="d"}) or on(x) (baz{x="y"} <= x{a="b"})"#, r#"(foo{a="b", c="d"} + bar{a="b", c="d"}) or on (x) (baz{a="b", x="y"} <= x{a="b", x="y"})"#)]
    #[test_case(r#"foo + (bar or baz{a="b"})"#, r#"foo + (bar or baz{a="b"})"#)]
    #[test_case(r#"foo + (bar{a="b"} or baz{a="b"})"#, r#"foo{a="b"} + (bar{a="b"} or baz{a="b"})"#)]
    #[test_case(r#"foo + (bar{a="b",c="d"} or baz{a="b"})"#, r#"foo{a="b"} + (bar{a="b", c="d"} or baz{a="b"})"#)]
    #[test_case(r#"foo{a="b"} + (bar OR baz{x="y"})"#, r#"foo{a="b"} + (bar{a="b"} or baz{a="b", x="y"})"#)]
    #[test_case(r#"foo{a="b"} + (bar{x="y",z="456"} OR baz{x="y",z="123"})"#, r#"foo{a="b", x="y"} + (bar{a="b", x="y", z="456"} or baz{a="b", x="y", z="123"})"#)]
    #[test_case(r#"foo{a="b"} unless bar{c="d"}"#, r#"foo{a="b"} unless bar{a="b", c="d"}"#)]
    #[test_case(r#"foo{a="b"} unless on() bar{c="d"}"#, r#"foo{a="b"} unless on () bar{c="d"}"#)]
    #[test_case(r#"foo + (bar{x="y"} unless baz{a="b"})"#, r#"foo{x="y"} + (bar{x="y"} unless baz{a="b", x="y"})"#)]
    #[test_case(r#"foo + (bar{x="y"} unless on() baz{a="b"})"#, r#"foo + (bar{x="y"} unless on () baz{a="b"})"#)]
    #[test_case(r#"foo{a="b"} + (bar UNLESS baz{x="y"})"#, r#"foo{a="b"} + (bar{a="b"} unless baz{a="b", x="y"})"#)]
    #[test_case(r#"foo{a="b"} + (bar{x="y"} unLESS baz)"#, r#"foo{a="b", x="y"} + (bar{a="b", x="y"} unless baz{a="b", x="y"})"#)]
    fn special_binary_expressions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // aggregate fns
    #[test_case(r#"sum(foo{bar="baz"}) / a{b="c"}"#, r#"sum(foo{bar="baz"}) / a{b="c"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) by () / a{b="c"}"#, r#"sum(foo{bar="baz"}) by () / a{b="c"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) by (bar) / a{b="c"}"#, r#"sum(foo{bar="baz"}) by (bar) / a{b="c", bar="baz"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) by (b) / a{b="c"}"#, r#"sum(foo{b="c", bar="baz"}) by (b) / a{b="c"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#, r#"sum(foo{bar="baz"}) by (x) / a{b="c"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) by (bar,b) / a{b="c"}"#, r#"sum(foo{b="c", bar="baz"}) by (bar, b) / a{b="c", bar="baz"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) without () / a{b="c"}"#, r#"sum(foo{b="c", bar="baz"}) without () / a{b="c", bar="baz"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) without (bar) / a{b="c"}"#, r#"sum(foo{b="c", bar="baz"}) without (bar) / a{b="c"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) without (b) / a{b="c"}"#, r#"sum(foo{bar="baz"}) without (b) / a{b="c", bar="baz"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) without (x) / a{b="c"}"#, r#"sum(foo{b="c", bar="baz"}) without (x) / a{b="c", bar="baz"}"#)]
    #[test_case(r#"sum(foo{bar="baz"}) without (bar,b) / a{b="c"}"#, r#"sum(foo{bar="baz"}) without (bar, b) / a{b="c"}"#)]
    #[test_case(r#"sum(foo, bar) by (a) + baz{a="b"}"#, r#"sum(foo{a="b"}, bar) by (a) + baz{a="b"}"#)]
    #[test_case(r#"topk(3, foo) by (baz,x) + bar{baz="a"}"#, r#"topk(3, foo{baz="a"}) by (baz, x) + bar{baz="a"}"#)]
    #[test_case(r#"topk(a, foo) without (x,y) + bar{baz="a"}"#, r#"topk(a, foo{baz="a"}) without (x, y) + bar{baz="a"}"#)]
    #[test_case(r#"a{b="c"} + quantiles("foo", 0.1, 0.2, bar{x="y"}) by (b, x, y)"#, r#"a{b="c", x="y"} + quantiles("foo", 0.1, 0.2, bar{b="c", x="y"}) by (b, x, y)"#)]
    #[test_case(r#"count_values("foo", bar{baz="a"}) by (bar,b) + a{b="c"}"#, r#"count_values("foo", bar{baz="a"}) by (bar, b) + a{b="c"}"#)]
    fn aggregate_functions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // unknown fn
    #[test_case(r#"f(foo) + bar{baz="a"}"#, r#"f(foo) + bar{baz="a"}"#)]
    #[test_case(r#"f(a,b,foo{a="b"} / bar) + baz{x="y"}"#, r#"f(a, b, foo{a="b"} / bar{a="b"}) + baz{x="y"}"#)]
    fn unknown_function(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // transform fns
    // #[test_case(r#"round(foo{bar = "baz"}) + sqrt(b{z=~"d"})"#, r#"round(foo{bar="baz", z=~"d"}) + sqrt(b{bar="baz", z=~"d"})"#)]
    // #[test_case(r#"foo{bar="baz"} + SQRT(a{z=~"c"})"#, r#"foo{bar="baz", z=~"c"} + SQRT(a{bar="baz", z=~"c"})"#)]
    #[test_case(r#"round({__name__="foo"}) + bar"#, r#"round(foo) + bar"#)]
    #[test_case(r#"round({__name__=~"foo|bar"}) + baz"#, r#"round({__name__=~"foo|bar"}) + baz"#)]
    #[test_case(r#"round({__name__=~"foo|bar",a="b"}) + baz"#, r#"round({__name__=~"foo|bar", a="b"}) + baz{a="b"}"#)]
    #[test_case(r#"round({__name__=~"foo|bar",a="b"}) + sqrt(baz)"#, r#"round({__name__=~"foo|bar", a="b"}) + sqrt(baz{a="b"})"#)]
    #[test_case(r#"round(foo) + {__name__="bar",x="y"}"#, r#"round(foo{x="y"}) + bar{x="y"}"#)]
    // #[test_case(r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#, r#"absent(foo{bar="baz"}) + sqrt(a{z=~"c"})"#)]
    #[test_case(r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#, r#"ABSENT(foo{bar="baz"}) + sqrt(a{z=~"c"})"#)]
    #[test_case(r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#, r#"label_set(foo{bar="baz"}, "xx", "y") + a{x="y"}"#)]
    #[test_case(r#"now() + foo{bar="baz"} + x{y="x"}"#, r#"(now() + foo{bar="baz", y="x"}) + x{bar="baz", y="x"}"#)]
    #[test_case(r#"limit_offset(5, 10, {x="y"}) if {a="b"}"#, r#"limit_offset(5, 10, {a="b", x="y"}) if {a="b", x="y"}"#)]
    #[test_case(r#"buckets_limit(aa, {x="y"}) if {a="b"}"#, r#"buckets_limit(aa, {a="b", x="y"}) if {a="b", x="y"}"#)]
    #[test_case(r#"histogram_quantiles("q", 0.1, 0.9, {x="y"}) - {a="b"}"#, r#"histogram_quantiles("q", 0.1, 0.9, {a="b", x="y"}) - {a="b", x="y"}"#)]
    #[test_case(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#, r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le)) - {a="b"}"#)]
    #[test_case(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x)) - {a="b"}"#, r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le, x)) - {a="b", x="y"}"#)]
    #[test_case(r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({x="y"}[5m])) by (le,x,a)) - {a="b"}"#, r#"histogram_quantiles("q", 0.1, 0.9, sum(rate({a="b", x="y"}[5m])) by (le, x, a)) - {a="b", x="y"}"#)]
    #[test_case(r#"vector(foo) + bar{a="b"}"#, r#"vector(foo) + bar{a="b"}"#)]
    #[test_case(r#"vector(foo{x="y"} + a) + bar{a="b"}"#, r#"vector(foo{x="y"} + a{x="y"}) + bar{a="b"}"#)]
    fn transform_functions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // multilevel transform fns
    #[test_case(r#"round(sqrt(foo)) + bar"#, r#"round(sqrt(foo)) + bar"#)]
    #[test_case(r#"round(sqrt(foo)) + bar{b="a"}"#, r#"round(sqrt(foo{b="a"})) + bar{b="a"}"#)]
    #[test_case(r#"round(sqrt(foo{a="b"})) + bar{x="y"}"#, r#"round(sqrt(foo{a="b", x="y"})) + bar{a="b", x="y"}"#)]
    fn multilevel_transform_functions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // rollup fns
    #[test_case(r#"RATE(foo[5m]) / rate(baz{a="b"}) + increase(x{y="z"} offset 5i)"#, r#"(RATE(foo{a="b", y="z"}[5m]) / rate(baz{a="b", y="z"})) + increase(x{a="b", y="z"} offset 5i)"#)]
    #[test_case(r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#, r#"sum(rate(foo[5m])) / rate(baz{a="b"})"#)]
    #[test_case(r#"sum(rate(foo[5m])) by (a) / rate(baz{a="b"})"#, r#"sum(rate(foo{a="b"}[5m])) by (a) / rate(baz{a="b"})"#)]
    #[test_case(r#"rate({__name__="foo"}) + rate({__name__="bar",x="y"}) - rate({__name__=~"baz"})"#, r#"(rate(foo{x="y"}) + rate(bar{x="y"})) - rate({__name__=~"baz", x="y"})"#)]
    #[test_case(r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz)"#, r#"rate({__name__=~"foo|bar", x="y"}) + rate(baz{x="y"})"#)]
    #[test_case(r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#, r#"absent_over_time(foo{x="y"}[5m]) + bar{a="b"}"#)]
    #[test_case(r#"{x="y"} + quantile_over_time(0.5, {a="b"})"#, r#"{a="b", x="y"} + quantile_over_time(0.5, {a="b", x="y"})"#)]
    #[test_case(r#"quantiles_over_time("quantile", 0.1, 0.9, foo{x="y"}[5m] offset 4h) + bar{a!="b"}"#, r#"quantiles_over_time("quantile", 0.1, 0.9, foo{a!="b", x="y"}[5m] offset 4h) + bar{a!="b", x="y"}"#)]
    fn rollup_functions(q: &str, expected: &str) {
        test_optimize(q, expected)
    }

    // @ modifier
    #[test_case(r#"foo @ end() + bar{baz="a"}"#, r#"foo{baz="a"} @ end() + bar{baz="a"}"#)]
    #[test_case(r#"sum(foo @ end()) + bar{baz="a"}"#, r#"sum(foo @ end()) + bar{baz="a"}"#)]
    #[test_case(r#"foo @ (bar{a="b"} + baz{x="y"})"#, r#"foo @ (bar{a="b", x="y"} + baz{a="b", x="y"})"#)]
    fn at_modifier(q: &str, expected: &str) {
        test_optimize(q, expected)
    }

    // subqueries
    #[test_case(r#"rate(avg_over_time(foo[5m:])) + bar{baz="a"}"#, r#"rate(avg_over_time(foo{baz="a"}[5m:])) + bar{baz="a"}"#)]
    #[test_case(r#"rate(sum(foo[5m:])) + bar{baz="a"}"#, r#"rate(sum(foo[5m:])) + bar{baz="a"}"#)]
    #[test_case(r#"rate(sum(foo[5m:]) by (baz)) + bar{baz="a"}"#, r#"rate(sum(foo{baz="a"}[5m:]) by (baz)) + bar{baz="a"}"#)]
    fn subqueries(q: &str, expected: &str) {
        test_optimize(q, expected)
    }


    // binary ops with constants or scalars
    #[test_case(r#"200 * foo / bar{baz="a"}"#, r#"(200 * foo{baz="a"}) / bar{baz="a"}"#)]
    #[test_case(r#"foo * 100 / bar{baz="a"}"#, r#"(foo{baz="a"} * 100) / bar{baz="a"}"#)]
    #[test_case(r#"foo / bar{baz="a"} * 100"#, r#"(foo{baz="a"} / bar{baz="a"}) * 100"#)]
    #[test_case(r#"scalar(x) * foo / bar{baz="a"}"#, r#"(scalar(x) * foo{baz="a"}) / bar{baz="a"}"#)]
   // #[test_case(r#"SCALAR(x) * foo / bar{baz="a"}"#, r#"(SCALAR(x) * foo{baz="a"}) / bar{baz="a"}"#)]
    #[test_case(r#"100 * on(foo) bar{baz="z"} + a"#, r#"(100 * on (foo) bar{baz="z"}) + a"#)]
    fn binary_ops_with_constants_or_scalars(q: &str, expected: &str) {
        test_optimize(q, expected)
    }
    
}