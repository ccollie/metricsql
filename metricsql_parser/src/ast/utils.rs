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

use crate::ast::{AggregationExpr, BinaryExpr, Expr, FunctionExpr, NumberLiteral, ParensExpr};
use crate::common::Operator;
use crate::prelude::{BinModifier, MetricExpr};

/// Create a selector expression based on a qualified or unqualified column name
///
/// example:
/// ```
/// use crate::metricsql_parser::ast::utils::*;
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
        Expr::Number(NumberLiteral { value, .. }) => *value == val,
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
    match expr {
        Expr::Number(NumberLiteral { value, .. }) => value.is_nan(),
        _ => false,
    }
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
/// # use crate::metricsql_parser::ast::utils::{selector, number, conjunction};
/// // a=1 AND b=2
/// let expr = selector("a").eq(number(1.0)).and(selector("b").eq(number(2.0)));
///
/// // [a=1, b=2]
/// let split = vec![
///   selector("a").eq(number(1.0)),
///   selector("b").eq(number(2.0)),
/// ];
///
/// // use conjunction to join them together with `AND`
/// assert_eq!(conjunction(split), Some(expr));
/// ```
pub fn conjunction(filters: impl IntoIterator<Item = Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.and(expr))
}

/// Combines an array of filter expressions into a single filter
/// expression consisting of the input filter expressions joined with
/// logical OR.
///
/// Returns None if the filters array is empty.
pub fn disjunction(filters: impl IntoIterator<Item = Expr>) -> Option<Expr> {
    filters.into_iter().reduce(|accum, expr| accum.or(expr))
}

// all this nonsense because f64 used in NumberExpr doesn't implement Eq
pub fn expr_equals(expr1: &Expr, expr2: &Expr) -> bool {
    use Expr::*;

    fn compare_parens(parens: &ParensExpr, expr: &Expr) -> bool {
        if let Some(other) = parens.innermost_expr() {
            return expr == other;
        }
        false
    }

    match (expr1, expr2) {
        (Aggregation(ae1), Aggregation(ae2)) => aggregation_exprs_equal(ae1, ae2),
        // special case: (x) == x. I don't know if i like this
        (Parens(p), e) => p.len() == 1 && compare_parens(p, e),
        (e, Parens(p)) => p.len() == 1 && compare_parens(p, e),
        (a, b) => a == b,
    }
}

fn aggregation_exprs_equal(ae1: &AggregationExpr, ae2: &AggregationExpr) -> bool {
    ae1.function == ae2.function
        && ae1.limit == ae2.limit
        && ae1.keep_metric_names == ae2.keep_metric_names
        && ae1.arg_idx_for_optimization == ae2.arg_idx_for_optimization
        && expr_vec_equals(&ae1.args, &ae2.args)
}

fn binary_exprs_equal(be1: &BinaryExpr, be2: &BinaryExpr) -> bool {
    be1.op == be2.op
        && expr_equals(&be1.left, &be2.left)
        && expr_equals(&be1.right, &be2.right)
        && bin_modifiers_equal(&be1.modifier, &be2.modifier)
}

fn bin_modifiers_equal(bm1: &Option<BinModifier>, bm2: &Option<BinModifier>) -> bool {
    match (bm1, bm2) {
        (Some(bm1), Some(bm2)) => bm1 == bm2,
        (None, None) => true,
        _ => {
            // None, Some
            let default_value = BinModifier::default();
            let left = bm1.as_ref().unwrap_or(&default_value);
            let right = bm2.as_ref().unwrap_or(&default_value);
            left == right
        }
    }
}

fn function_exprs_equal(fe1: &FunctionExpr, fe2: &FunctionExpr) -> bool {
    fe1.name == fe2.name
        && fe1.keep_metric_names == fe2.keep_metric_names
        && fe1.arg_idx_for_optimization == fe2.arg_idx_for_optimization
        && fe1.is_scalar == fe2.is_scalar
        && fe1.return_type == fe2.return_type
        && expr_vec_equals(&fe1.args, &fe2.args)
}

fn expr_vec_equals(exprs1: &Vec<Expr>, exprs2: &Vec<Expr>) -> bool {
    if exprs1.len() != exprs2.len() {
        return false;
    }
    exprs1
        .iter()
        .zip(exprs2.iter())
        .all(|(e1, e2)| expr_equals(e1, e2))
}

#[cfg(test)]
pub mod tests {
    use crate::ast::utils::{conjunction, disjunction, selector};

    #[test]
    fn test_conjunction_empty() {
        assert_eq!(conjunction(vec![]), None);
    }

    #[test]
    fn test_conjunction() {
        // `[A, B, C]`
        let expr = conjunction(vec![selector("a"), selector("b"), selector("c")]);

        // --> `(A AND B) AND C`
        assert_eq!(
            expr,
            Some(selector("a").and(selector("b")).and(selector("c")))
        );

        // which is different than `A AND (B AND C)`
        assert_ne!(
            expr,
            Some(selector("a").and(selector("b").and(selector("c"))))
        );
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
        assert_eq!(
            expr,
            Some(selector("a").or(selector("b")).or(selector("c")))
        );

        // which is different than `A OR (B OR C)`
        assert_ne!(
            expr,
            Some(selector("a").or(selector("b").or(selector("c"))))
        );
    }
}
