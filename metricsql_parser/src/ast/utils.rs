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

use ahash::AHashSet;

use crate::ast::{BinaryExpr, Expr, MetricExpr, NumberLiteral, Operator, ParensExpr};

/// Create a selector expression based on a qualified or unqualified column name
///
/// example:
/// ``` rust
/// use crate::metricsql_parser::ast::utils::*;
/// let c = selector("latency");
/// ```
pub fn selector(ident: impl Into<String>) -> Expr {
    Expr::MetricExpression(MetricExpr::new(ident.into()))
}

pub fn lit(str: &str) -> Expr {
    Expr::from(str)
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
        Expr::NumberLiteral(NumberLiteral { value, .. }) => *value == val,
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
        Expr::NumberLiteral(NumberLiteral { value, .. }) => value.is_nan(),
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
/// ``` rust
/// use crate::metricsql_parser::ast::utils::{selector, number, conjunction};
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
        match expr {
            Parens(p) => parens == p,
            _ => false,
        }
    }

    match (expr1, expr2) {
        // special case: (x) == x. I don't know if i like this
        (Parens(p), e) => p.len() == 1 && compare_parens(p, e),
        (e, Parens(p)) => p.len() == 1 && compare_parens(p, e),
        (a, b) => a == b,
    }
}

pub(super) fn string_vecs_equal_unordered(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let hash_a: AHashSet<_> = a.iter().collect();
    b.iter().all(|x| hash_a.contains(x))
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
