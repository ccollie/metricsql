// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Utility functions for expression simplification

use crate::common::Operator;
use crate::ast::{AggregationExpr, BExpr, BinaryExpr, Expr, FunctionExpr, NumberExpr, RollupExpr, WithArgExpr, WithExpr};
use crate::prelude::MetricExpr;


/// Create a selector expression based on a qualified or unqualified column name
///
/// example:
/// ```
/// use crate::hir::*;
/// let c = selector("latency");
/// ```
pub fn selector(ident: impl Into<String>) -> Expr {
    Expr::MetricExpression(MetricExpr::new(ident.into()))
}

pub fn lit(str: &str) -> Expr {
    Expr::StringLiteral(str.to_string())
}

pub fn number(val: f64) -> Expr {
    Expr::from(val)
}

/// returns true if `needle` is found in a chain of search_op
/// expressions. Such as: (A AND B) AND C
pub fn expr_contains(expr: &Expr, needle: &Expr, search_op: Operator) -> bool {
    match expr {
        Expr::BinaryOperator(BinaryExpr {
                                 left, op, right, ..
                             }) if *op == search_op => {
            expr_contains(left, needle, search_op) || expr_contains(right, needle, search_op)
        }
        _ => expr == needle,
    }
}

pub fn is_number_value(s: &Expr, val: f64) -> bool {
    match s {
        Expr::Number(NumberExpr { value, .. }) => *value == val,
        _ => false,
    }
}

pub fn is_zero(s: &Expr) -> bool {
    is_number_value(s, 0.0)
}

pub fn is_one(s: &Expr) -> bool {
    is_number_value(s, 1.0)
}

pub fn is_null(expr: &Expr) -> bool {
    is_number_value(expr, f64::NAN)
}

/// returns true if `haystack` looks like (needle OP X) or (X OP needle)
pub fn is_op_with(target_op: Operator, haystack: &Expr, needle: &Expr) -> bool {
    matches!(haystack, Expr::BinaryOperator(BinaryExpr { left, op, right, .. })
        if op == &target_op && (needle == left.as_ref() || needle == right.as_ref()))
}

/// Combines an array of filter expressions into a single filter
/// expression consisting of the input filter expressions joined with
/// logical AND.
///
/// Returns None if the filters array is empty.
///
/// # Example
/// ```
/// # use crate::hir::expr_fn::{selector, number};
/// # use crate::hir::utils::conjunction;
/// // a=1 AND b=2
/// let expr = selector("a").eq(number(1)).and(selector("b").eq(number(2)));
///
/// // [a=1, b=2]
/// let split = vec![
///   selector("a").eq(number(1)),
///   selector("b").eq(number(2)),
/// ];
///
/// // use conjunction to join them together with `AND`
/// assert_eq!(conjunction(split), Some(expr));
/// ```
pub fn conjunction(filters: impl IntoIterator<Item =Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.and(expr))
}

/// Combines an array of filter expressions into a single filter
/// expression consisting of the input filter expressions joined with
/// logical OR.
///
/// Returns None if the filters array is empty.
pub fn disjunction(filters: impl IntoIterator<Item =Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.or(expr))
}

// all this nonsense because f64 used in NumberExpr doesn't implement Eq
pub fn expr_equals(expr1: &Expr, expr2: &Expr) -> bool {
    use Expr::*;

    match (expr1, expr2) {
        (Duration(d1), Duration(d2)) => d1.value == d2.value,
        (MetricExpression(me1), MetricExpression(me2)) => me1 == me2,
        (StringLiteral(s1), StringLiteral(s2)) => s1 == s2,
        (Number(n1), Number(n2)) => {
            (n1.value - n2.value).abs() < f64::EPSILON
        },
        (BinaryOperator(be1), BinaryOperator(be2)) => binary_exprs_equal(be1, be2),
        (Function(fe1), Function(fe2)) => function_exprs_equal(fe1, fe2),
        (Aggregation(ae1), Aggregation(ae2)) => aggregation_exprs_equal(ae1, ae2),
        (Parens(pe1), Parens(pe2)) => expr_vec_equals(&pe1.expressions, &pe2.expressions),
        (Rollup(re1), Rollup(re2)) =>  rollup_exprs_equal(re1, re2),
        (With(we1), With(we2)) => with_exprs_equal(we1, we2),
        (StringExpr(s1), StringExpr(s2)) => s1 == s2,
        _ => false
    }
}

fn optional_exprs_equal(e1: &Option<Expr>, e2: &Option<Expr>) -> bool {
    match (e1, e2) {
        (Some(e1), Some(e2)) => expr_equals(e1, e2),
        (None, None) => true,
        _ => false,
    }
}

fn optional_boxed_exprs_equal(e1: &Option<BExpr>, e2: &Option<BExpr>) -> bool {
    match (e1, e2) {
        (Some(e1), Some(e2)) => expr_equals(e1, e2),
        (None, None) => true,
        _ => false,
    }
}

fn rollup_exprs_equal(re1: &RollupExpr, re2: &RollupExpr) -> bool {
    re1.inherit_step == re2.inherit_step &&
        re1.window == re2.window &&
        re1.step == re2.step &&
        re1.offset == re2.offset &&
        expr_equals(&re1.expr, &re2.expr) &&
        optional_boxed_exprs_equal(&re1.at, &re2.at)
}

fn aggregation_exprs_equal(ae1: &AggregationExpr, ae2: &AggregationExpr) -> bool {
    ae1.name == ae2.name &&
        ae1.limit == ae2.limit &&
        ae1.keep_metric_names == ae2.keep_metric_names &&
        ae1.arg_idx_for_optimization ==  ae2.arg_idx_for_optimization &&
        expr_vec_equals(&ae1.args, &ae2.args)
}

fn binary_exprs_equal(be1: &BinaryExpr, be2: &BinaryExpr) -> bool {
    be1.op == be2.op &&
        be1.bool_modifier == be2.bool_modifier &&
        be1.group_modifier == be2.group_modifier &&
        be1.join_modifier == be2.join_modifier &&
        expr_equals(&be1.left, &be2.left) &&
        expr_equals(&be1.right, &be2.right) &&
        be1.modifier == be2.modifier
}

fn function_exprs_equal(fe1: &FunctionExpr, fe2: &FunctionExpr) -> bool {
    fe1.name == fe2.name &&
        fe1.keep_metric_names == fe2.keep_metric_names &&
        fe1.arg_idx_for_optimization == fe2.arg_idx_for_optimization &&
        fe1.is_scalar == fe2.is_scalar &&
        fe1.return_type == fe2.return_type &&
        expr_vec_equals(&fe1.args, &fe2.args)
}

fn with_exprs_equal(we1: &WithExpr, we2: &WithExpr) -> bool {
    expr_equals(&we1.expr, &we2.expr) &&
    we1.was.len() == we2.was.len() &&
        we1.was.iter().zip(we2.was.iter()).all(|(w1, w2)| with_arg_exprs_equal(w1, w2))
}

fn with_arg_exprs_equal(we1: &WithArgExpr, we2: &WithArgExpr) -> bool {
    we1.name == we2.name &&
        we1.args == we2.args &&
        we1.is_function == we2.is_function &&
        expr_equals(&we1.expr, &we2.expr)
}

fn expr_vec_equals(exprs1: &[Expr], exprs2: &[Expr]) -> bool {
    if exprs1.len() != exprs2.len() {
        return false;
    }
    for (e1, e2) in exprs1.iter().zip(exprs2.iter()) {
        if !expr_equals(e1, e2) {
            return false;
        }
    }
    true
}

// todo: use expr_visitor instead
pub fn visit_all(e: &mut Expr, visitor: fn(&mut Expr) -> ()) {
    match e {
        Expr::BinaryOperator(be) => {
            visit_all(&mut be.left, visitor);
            visit_all(&mut be.right, visitor);
        }
        Expr::Function(fe) => {
            for arg in fe.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::Aggregation(ae) => {
            for arg in ae.args.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::With(we) => {
            visit_all(&mut we.expr, visitor);
            for wa in we.was.iter_mut() {
                visit_all(&mut wa.expr, visitor)
            }
        }
        Expr::Parens(pe) => {
            for arg in pe.expressions.iter_mut() {
                visit_all(arg, visitor)
            }
        }
        Expr::Rollup(re) => {
            visit_all(&mut re.expr, visitor);
            if let Some(at) = &mut re.at {
                visit_all(at, visitor)
            }
        },
        _ => {}
    }
    visitor(e);
}


#[cfg(test)]
pub mod tests {
    use crate::ast::utils::{selector, conjunction, disjunction};

    #[test]
    fn test_conjunction_empty() {
        assert_eq!(conjunction(vec![]), None);
    }

    #[test]
    fn test_conjunction() {
        // `[A, B, C]`
        let expr = conjunction(vec![selector("a"), selector("b"), selector("c")]);

        // --> `(A AND B) AND C`
        assert_eq!(expr, Some(selector("a").and(selector("b")).and(selector("c"))));

        // which is different than `A AND (B AND C)`
        assert_ne!(expr, Some(selector("a").and(selector("b").and(selector("c")))));
    }

    #[test]
    fn test_disjunction_empty() {
        assert_eq!(disjunction(vec![]), None);
    }

    #[test]
    fn test_disjunction() {
        // `[A, B, C]`
        let expr = disjunction(vec![selector("a"), selector("b"), selector("c")]);

        // --> `(A OR B) OR C`
        assert_eq!(expr, Some(selector("a").or(selector("b")).or(selector("c"))));

        // which is different than `A OR (B OR C)`
        assert_ne!(expr, Some(selector("a").or(selector("b").or(selector("c")))));
    }
}
